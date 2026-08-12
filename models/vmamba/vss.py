"""
SS2D + VSSBlock, trimmed from VMamba's vmamba.py.

WHAT WAS REMOVED AND WHY
  - SS2Dv0 / SS2Dv3 / SS2Dm0 forward variants. We use forward_type="v05",
    which lives in SS2Dv2. SS2Dm0 is the only consumer of
    mamba2/ssd_minimal.py, whose bare try/except in upstream RE-RAISES and
    therefore makes the whole module unimportable without that folder. This
    is the blocker that is now gone.
  - VSSM, Backbone_VSSM, PatchMerging2D and the vmamba_* factories: we build
    our own 3-block model on a shared stem, so the multi-stage backbone and
    downsampling machinery is dead weight.
  - VSSM.flops() and the fvcore import: fvcore is not on Colab, and the flop
    counter needs a SelectiveScanCuda op registration that does not exist on
    the torch fallback path. Report params + measured img/s instead.
  - timm's DropPath and trunc_normal_: we run drop_path=0 (identity), and
    torch.nn.init.trunc_normal_ is in stock PyTorch. One fewer dependency.

SS2Dv2.__initv2__, forward_corev2 and forwardv2 are otherwise UNMODIFIED,
as is mamba_init. The S4D initialisation of A_log/D/dt_proj IS the mechanism
under test and must not be replaced by our generic init.
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .csm_triton import cross_scan_fn, cross_merge_fn
from .csms6s import selective_scan_fn


class Linear2d(nn.Linear):
    def forward(self, x):
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random",
                dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device
                         ).view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init,
                    dt_min, dt_max, dt_init_floor, k_group=4):
        dt_projs = [cls.dt_init(dt_rank, d_inner, dt_scale, dt_init,
                                dt_min, dt_max, dt_init_floor)
                    for _ in range(k_group)]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))
        del dt_projs
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True)
        return A_logs, Ds, dt_projs_weight, dt_projs_bias


class SS2D(nn.Module):
    """2D selective scan. Upstream SS2Dv2 path only (forward_type v0x)."""

    def __init__(self, d_model=96, d_state=16, ssm_ratio=2.0, dt_rank="auto",
                 act_layer=nn.SiLU, d_conv=3, conv_bias=True, dropout=0.0,
                 bias=False, dt_min=0.001, dt_max=0.1, dt_init="random",
                 dt_scale=1.0, dt_init_floor=1e-4, initialize="v0",
                 forward_type="v05", channel_first=False,
                 force_torch_scan=True, **kwargs):
        super().__init__()
        self.k_group = 4
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        self.force_torch_scan = force_torch_scan
        Linear = Linear2d if channel_first else nn.Linear

        self.disable_force32, forward_type = self._checkpostfix("_no32", forward_type)
        self.oact, forward_type = self._checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = self._checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = self._checkpostfix("_nozact", forward_type)
        self.out_norm, forward_type = self._get_outnorm(forward_type, self.d_inner, channel_first)

        FORWARD_TYPES = dict(
            v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32),
                        selective_scan_backend="mamba"),
            v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32),
                        selective_scan_backend="oflex"),
            v04=partial(self.forward_corev2, force_fp32=False),
            v05=partial(self.forward_corev2, force_fp32=False, no_einsum=True),
            v051d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="unidi"),
            v052d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="bidi"),
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32),
                       selective_scan_backend="core"),
            v3=partial(self.forward_corev2, force_fp32=False, selective_scan_backend="oflex"),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)
        if self.forward_core is None:
            raise ValueError(f"unsupported forward_type {forward_type!r} in trimmed SS2D")

        d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
        self.in_proj = Linear(self.d_model, d_proj, bias=bias)
        self.act = act_layer()

        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner, out_channels=self.d_inner,
                groups=self.d_inner, bias=conv_bias, kernel_size=d_conv,
                padding=(d_conv - 1) // 2)

        x_proj = [nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
                  for _ in range(self.k_group)]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in x_proj], dim=0))
        del x_proj

        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = \
                mamba_init.init_dt_A_D(self.d_state, self.dt_rank, self.d_inner,
                                       dt_scale, dt_init, dt_min, dt_max,
                                       dt_init_floor, k_group=self.k_group)
        elif initialize in ["v1"]:
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.randn((self.k_group * self.d_inner, self.d_state)))
            self.dt_projs_weight = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner)))
        elif initialize in ["v2"]:
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.zeros((self.k_group * self.d_inner, self.d_state)))
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.k_group, self.d_inner)))

        # The S4D init above IS the Mamba mechanism. Protect it from the
        # generic trunc_normal_ init applied by MotivationNet.
        for m in self.modules():
            m._skip_generic_init = True

    @staticmethod
    def _checkpostfix(tag, value):
        ret = value[-len(tag):] == tag
        if ret:
            value = value[:-len(tag)]
        return ret, value

    @staticmethod
    def _get_outnorm(forward_type="", d_inner=192, channel_first=True):
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value
        LN = LayerNorm2d if channel_first else nn.LayerNorm
        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)
        if out_norm_none:
            out_norm = nn.Identity()
        elif out_norm_cnorm:
            out_norm = nn.Sequential(
                LN(d_inner),
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)))
        elif out_norm_dwconv3:
            out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)))
        elif out_norm_sigmoid:
            out_norm = nn.Sigmoid()
        else:
            out_norm = LN(d_inner)
        return out_norm, forward_type

    def forward_corev2(self, x=None, force_fp32=False, ssoflex=True,
                       no_einsum=False, selective_scan_backend=None,
                       scan_mode="cross2d", scan_force_torch=False, **kwargs):
        assert selective_scan_backend in [None, "oflex", "mamba", "torch"]
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None) \
            if isinstance(scan_mode, str) else scan_mode
        assert isinstance(_scan_mode, int)
        delta_softplus = True
        out_norm = self.out_norm
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
        scan_force_torch = scan_force_torch or self.force_torch_scan

        B, D, H, W = x.shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias,
                                     delta_softplus, ssoflex,
                                     backend=selective_scan_backend)

        x_proj_bias = getattr(self, "x_proj_bias", None)
        xs = cross_scan_fn(x, in_channel_first=True, out_channel_first=True,
                           scans=_scan_mode, force_torch=scan_force_torch)
        if no_einsum:
            x_dbl = F.conv1d(xs.view(B, -1, L), self.x_proj_weight.view(-1, D, 1),
                             bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None),
                             groups=K)
            dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts = F.conv1d(dts.contiguous().view(B, -1, L),
                               self.dt_projs_weight.view(K * D, -1, 1), groups=K)
        else:
            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
            if x_proj_bias is not None:
                x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
            dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        As = -self.A_logs.to(torch.float).exp()
        Ds = self.Ds.to(torch.float)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        ys = selective_scan(xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
                            ).view(B, K, -1, H, W)
        y = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True,
                           scans=_scan_mode, force_torch=scan_force_torch)

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = out_norm(y)
        return y.to(x.dtype)

    def forward(self, x):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
            if not self.disable_z_act:
                z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x = self.conv2d(x)
        x = self.act(x)
        y = self.forward_core(x)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        return self.dropout(self.out_proj(y))


class VSSBlock(nn.Module):
    """Pre-norm VSS block: x + SS2D(LN(x)); x + MLP(LN2(x)). DropPath removed."""

    def __init__(self, hidden_dim=0, drop_path=0., norm_layer=nn.LayerNorm,
                 channel_first=False, ssm_d_state=16, ssm_ratio=2.0,
                 ssm_dt_rank="auto", ssm_act_layer=nn.SiLU, ssm_conv=3,
                 ssm_conv_bias=True, ssm_drop_rate=0., ssm_init="v0",
                 forward_type="v05", mlp_ratio=4.0, mlp_act_layer=nn.GELU,
                 mlp_drop_rate=0.0, post_norm=False, force_torch_scan=True,
                 **kwargs):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.post_norm = post_norm
        if drop_path != 0:
            raise ValueError("drop_path must be 0: DropPath was removed with timm")

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SS2D(d_model=hidden_dim, d_state=ssm_d_state,
                           ssm_ratio=ssm_ratio, dt_rank=ssm_dt_rank,
                           act_layer=ssm_act_layer, d_conv=ssm_conv,
                           conv_bias=ssm_conv_bias, dropout=ssm_drop_rate,
                           initialize=ssm_init, forward_type=forward_type,
                           channel_first=channel_first,
                           force_torch_scan=force_torch_scan)
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            self.mlp = Mlp(in_features=hidden_dim,
                           hidden_features=int(hidden_dim * mlp_ratio),
                           act_layer=mlp_act_layer, drop=mlp_drop_rate,
                           channels_first=channel_first)

    def forward(self, x):
        if self.ssm_branch:
            x = x + (self.norm(self.op(x)) if self.post_norm else self.op(self.norm(x)))
        if self.mlp_branch:
            x = x + (self.norm2(self.mlp(x)) if self.post_norm else self.mlp(self.norm2(x)))
        return x