"""
Selective scan, trimmed from VMamba's csms6s.py.

WHAT WAS REMOVED AND WHY
  - fvcore flop-counting helpers (flops_selective_scan_fn/_ref,
    selective_scan_flop_jit): they import fvcore, which is not on Colab, and
    they only feed VSSM.flops(), which we do not use.
  - The __main__ benchmark block.
  - The SelectiveScanCuda autograd.Function is now defined ONLY IF a CUDA
    extension actually imported. Upstream defines it unconditionally, and its
    @torch.cuda.amp.custom_fwd / custom_bwd decorators are applied at class
    definition time. That API is deprecated in torch >= 2.4, so on a recent
    torch the mere import can warn or fail even though the class is never
    called.

selective_scan_torch is UNMODIFIED. It is the reference implementation; the
CUDA kernels compute the same mathematics faster. On Colab none of
selective_scan_cuda_oflex / _core / selective_scan_cuda are built, so
WITH_CUDA is False and every call routes here automatically.
"""

import torch
import warnings

WITH_SELECTIVESCAN_OFLEX = True
WITH_SELECTIVESCAN_CORE = False
WITH_SELECTIVESCAN_MAMBA = True
try:
    import selective_scan_cuda_oflex
except ImportError:
    WITH_SELECTIVESCAN_OFLEX = False
try:
    import selective_scan_cuda_core
except ImportError:
    WITH_SELECTIVESCAN_CORE = False
try:
    import selective_scan_cuda
except ImportError:
    WITH_SELECTIVESCAN_MAMBA = False

WITH_CUDA = (WITH_SELECTIVESCAN_OFLEX or WITH_SELECTIVESCAN_CORE
             or WITH_SELECTIVESCAN_MAMBA)


def selective_scan_torch(
    u: torch.Tensor,        # (B, K * C, L)
    delta: torch.Tensor,    # (B, K * C, L)
    A: torch.Tensor,        # (K * C, N)
    B: torch.Tensor,        # (B, K, N, L)
    C: torch.Tensor,        # (B, K, N, L)
    D: torch.Tensor = None,
    delta_bias: torch.Tensor = None,
    delta_softplus=True,
    oflex=True,
    *args,
    **kwargs
):
    dtype_in = u.dtype
    Batch, K, N, L = B.shape
    KCdim = u.shape[1]
    Cdim = int(KCdim / K)
    assert u.shape == (Batch, KCdim, L)
    assert delta.shape == (Batch, KCdim, L)
    assert A.shape == (KCdim, N)
    assert C.shape == B.shape

    if delta_bias is not None:
        delta = delta + delta_bias[..., None]
    if delta_softplus:
        delta = torch.nn.functional.softplus(delta)

    u, delta, A, B, C = u.float(), delta.float(), A.float(), B.float(), C.float()
    B = B.view(Batch, K, 1, N, L).repeat(1, 1, Cdim, 1, 1).view(Batch, KCdim, N, L)
    C = C.view(Batch, K, 1, N, L).repeat(1, 1, Cdim, 1, 1).view(Batch, KCdim, N, L)
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)

    x = A.new_zeros((Batch, KCdim, N))
    ys = []
    for i in range(L):
        x = deltaA[:, :, i, :] * x + deltaB_u[:, :, i, :]
        y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        ys.append(y)
    y = torch.stack(ys, dim=2)

    out = y if D is None else y + u * D.unsqueeze(-1)
    return out if oflex else out.to(dtype=dtype_in)


if WITH_CUDA:
    class SelectiveScanCuda(torch.autograd.Function):
        @staticmethod
        @torch.cuda.amp.custom_fwd
        def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None,
                    delta_softplus=False, oflex=True, backend=None):
            ctx.delta_softplus = delta_softplus
            backend = "oflex" if WITH_SELECTIVESCAN_OFLEX and (backend is None) else backend
            backend = "core" if WITH_SELECTIVESCAN_CORE and (backend is None) else backend
            backend = "mamba" if WITH_SELECTIVESCAN_MAMBA and (backend is None) else backend
            ctx.backend = backend
            if backend == "oflex":
                out, x, *rest = selective_scan_cuda_oflex.fwd(
                    u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
            elif backend == "core":
                out, x, *rest = selective_scan_cuda_core.fwd(
                    u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
            elif backend == "mamba":
                out, x, *rest = selective_scan_cuda.fwd(
                    u, delta, A, B, C, D, None, delta_bias, delta_softplus)
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out

        @staticmethod
        @torch.cuda.amp.custom_bwd
        def backward(ctx, dout, *args):
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            backend = ctx.backend
            if dout.stride(-1) != 1:
                dout = dout.contiguous()
            if backend == "oflex":
                du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
                    u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1)
            elif backend == "core":
                du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
                    u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1)
            elif backend == "mamba":
                du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                    u, delta, A, B, C, D, None, delta_bias, dout, x, None, None,
                    ctx.delta_softplus, False)
            return du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None


def selective_scan_fn(u, delta, A, B, C, D=None, delta_bias=None,
                      delta_softplus=True, oflex=True, backend=None):
    fn = (selective_scan_torch if backend == "torch" or (not WITH_CUDA)
          else SelectiveScanCuda.apply)
    return fn(u, delta, A, B, C, D, delta_bias, delta_softplus, oflex, backend)