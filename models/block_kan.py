"""
KAN-Mixer block.

WHY MIXER-STYLE AND NOT PLAIN STACKED KAN LAYERS
The obvious KAN block -- LayerNorm -> KANLinear -> residual -- is CHANNEL
MIXING ONLY. Applied to a token sequence it cannot move information between
spatial positions. ViT (attention) and VSS (2D selective scan) both mix
tokens. Comparing a spatial model against a non-spatial one and "finding"
they differ would be a confound, not a result, and it is the first thing a
reviewer would catch.

So this block does both, matching the token-mixer + channel-mixer
meta-architecture that ViT and VSS also follow:

    x = x + KAN_token(LN(x))     across the 64 tokens
    x = x + KAN_channel(LN(x))   across the 192 dims

KNOWN ASYMMETRY, DISCLOSE IT
ViT and VSS both use a vanilla MLP for channel mixing (295,872 params, ~65%
of each block). This block uses a KAN for channel mixing instead. So ViT and
VSS share a large identical component that KAN does not have. The bias runs
toward INFLATING ViT<->VSS similarity and DEFLATING both KAN pairs -- i.e.
toward the conclusion we want, which is exactly the direction a sharp
reviewer looks for. We keep it because it matches how KAN is actually
deployed in the literature and in the target segmentation network, but the
paper must say so in one sentence.

Two implementation constraints from efficient_kan.py:
  - KANLinear.forward asserts x.dim() == 2, hence the flatten/reshape.
  - grid_range must be (-2, 2), not the (-1, 1) default. See config.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .efficient_kan import KANLinear

_ACT = {"silu": nn.SiLU, "gelu": nn.GELU, "relu": nn.ReLU}


def _apply_kan(layer: KANLinear, x: torch.Tensor) -> torch.Tensor:
    """Apply a KANLinear over the last dim of a 3-d tensor.

    KANLinear only accepts (batch, features), so fold the leading dims.
    """
    b, n, d = x.shape
    return layer(x.reshape(b * n, d)).reshape(b, n, -1)


class KANMixerBlock(nn.Module):
    """Token-mixing KAN + channel-mixing KAN, both pre-norm with residuals."""

    def __init__(self, embed_dim=192, n_tokens=64, grid_size=5, spline_order=3,
                 grid_range=(-2.0, 2.0), base_activation="silu",
                 token_mix=True, layer_norm_eps=1e-6):
        super().__init__()
        self.token_mix = token_mix
        act = _ACT[base_activation]
        kan_kw = dict(grid_size=grid_size, spline_order=spline_order,
                      base_activation=act, grid_range=list(grid_range))

        if token_mix:
            self.norm_token = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
            self.kan_token = KANLinear(n_tokens, n_tokens, **kan_kw)
        self.norm_channel = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.kan_channel = KANLinear(embed_dim, embed_dim, **kan_kw)

        # The shared init in stem.py must not overwrite the B-spline init:
        # the spline parameterisation IS the mechanism under test.
        for m in self.modules():
            if isinstance(m, KANLinear):
                m._skip_generic_init = True
                for p in m.parameters(recurse=False):
                    p._skip_generic_init = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, D) -> (B, N, D)."""
        if self.token_mix:
            # transpose so the KAN acts across tokens, not channels
            h = self.norm_token(x).transpose(1, 2)      # (B, D, N)
            x = x + _apply_kan(self.kan_token, h).transpose(1, 2)
        x = x + _apply_kan(self.kan_channel, self.norm_channel(x))
        return x

    @torch.no_grad()
    def grid_coverage(self, x: torch.Tensor) -> dict:
        """Fraction of pre-KAN activations falling inside the spline grid.

        DIAGNOSTIC, run this after one epoch. If coverage is low the spline
        branch is inactive for those inputs and KANLinear collapses toward
        SiLU + Linear, i.e. an ordinary MLP -- at which point the experiment
        is comparing an MLP to a Transformer while calling it KAN. Want >0.90.
        """
        out = {}
        if self.token_mix:
            g = self.kan_token.grid
            lo, hi = g[:, self.kan_token.spline_order].min(), g[:, -self.kan_token.spline_order - 1].max()
            h = self.norm_token(x).transpose(1, 2)
            out["token"] = float(((h >= lo) & (h <= hi)).float().mean())
        g = self.kan_channel.grid
        lo, hi = g[:, self.kan_channel.spline_order].min(), g[:, -self.kan_channel.spline_order - 1].max()
        h = self.norm_channel(x)
        out["channel"] = float(((h >= lo) & (h <= hi)).float().mean())
        return out


def make_kan_block(cfg: dict):
    """Returns a block_fn(index) -> KANMixerBlock for MotivationNet."""
    def block_fn(_idx: int) -> KANMixerBlock:
        return KANMixerBlock(
            embed_dim=cfg["embed_dim"],
            n_tokens=cfg["n_tokens"],
            grid_size=cfg["grid_size"],
            spline_order=cfg["spline_order"],
            grid_range=cfg["grid_range"],
            base_activation=cfg["base_activation"],
            token_mix=cfg["token_mix"],
            layer_norm_eps=cfg.get("layer_norm_eps", 1e-6),
        )
    return block_fn