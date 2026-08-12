"""
VSS block adapter.

VSSBlock expects a 2D grid, (B, H, W, C) with channel_first=False. Our stem
produces a token sequence, (B, 64, 192). This adapter reshapes 64 -> 8x8 on
the way in and flattens on the way out.

WHY THIS IS NOT A CONFOUND
All three blocks receive an identical (B, 64, 192) tensor at the boundary and
return an identical (B, 64, 192) tensor. The 2D reshape is INTERNAL to the
mechanism, and operating on a 2D grid is precisely what distinguishes the
VMamba scan from 1D attention -- it is the variable under test, not a
difference in the harness. Disclose it in one sentence.

WHY THIS EXACT BLOCK
Whatever VSSBlock configuration the target segmentation network ships, the
motivation study must use the same one. If the paper's model runs SS2D and
the motivation study ran a generic 1D bidirectional Mamba, a reviewer can say
the motivation measured a different architecture from the one that was built,
and that objection is unanswerable.

TOKEN COUNT LIMITATION, STATE IT IN THE PAPER
patch=4 on 32x32 gives an 8x8 grid = 64 positions. VMamba was designed for
56x56 at stage 1. Over 64 positions the SSM's long-range behaviour has little
room to manifest, so KAN/ViT/VSS differences here may be UNDERSTATED. Going
to patch=2 (256 tokens) would fix it for ViT and VSS but the KAN token-mixer
is KANLinear(N->N) at 10*N^2 params, which at N=256 is 655,360 and destroys
parity. So 64 tokens stays, and the resolution is named as a limitation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .vmamba.vss import VSSBlock

_ACT = {"silu": nn.SiLU, "gelu": nn.GELU, "relu": nn.ReLU}


class VSSTokenBlock(nn.Module):
    """(B, N, D) -> reshape to (B, g, g, D) -> VSSBlock -> (B, N, D)."""

    def __init__(self, embed_dim=192, grid=8, ssm_d_state=16, ssm_ratio=1.0,
                 ssm_conv=3, ssm_dt_rank="auto", mlp_ratio=4.0,
                 forward_type="v05", drop_path=0.0, ssm_drop_rate=0.0,
                 mlp_drop_rate=0.0, post_norm=False, force_torch_scan=True,
                 layer_norm_eps=1e-6, **kwargs):
        super().__init__()
        self.grid = grid
        self.block = VSSBlock(
            hidden_dim=embed_dim,
            drop_path=drop_path,
            norm_layer=lambda d: nn.LayerNorm(d, eps=layer_norm_eps),
            channel_first=False,
            ssm_d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_dt_rank=ssm_dt_rank,
            ssm_act_layer=_ACT["silu"],
            ssm_conv=ssm_conv,
            ssm_conv_bias=True,
            ssm_drop_rate=ssm_drop_rate,
            ssm_init="v0",
            forward_type=forward_type,
            mlp_ratio=mlp_ratio,
            mlp_act_layer=_ACT["gelu"],
            mlp_drop_rate=mlp_drop_rate,
            post_norm=post_norm,
            force_torch_scan=force_torch_scan,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        g = self.grid
        if n != g * g:
            raise ValueError(f"expected {g*g} tokens for a {g}x{g} grid, got {n}")
        x = x.view(b, g, g, d)
        x = self.block(x)
        return x.reshape(b, n, d)


def make_vss_block(cfg: dict):
    """Returns a block_fn(index) -> VSSTokenBlock for MotivationNet."""
    def block_fn(_idx: int) -> VSSTokenBlock:
        return VSSTokenBlock(
            embed_dim=cfg["embed_dim"],
            grid=cfg["grid"],
            ssm_d_state=cfg["ssm_d_state"],
            ssm_ratio=cfg["ssm_ratio"],
            ssm_conv=cfg["ssm_conv"],
            ssm_dt_rank=cfg["ssm_dt_rank"],
            mlp_ratio=cfg["mlp_ratio"],
            forward_type=cfg["forward_type"],
            drop_path=cfg["drop_path"],
            ssm_drop_rate=cfg["ssm_drop_rate"],
            mlp_drop_rate=cfg["mlp_drop_rate"],
            post_norm=cfg["post_norm"],
            force_torch_scan=cfg["force_torch_scan"],
            layer_norm_eps=cfg.get("layer_norm_eps", 1e-6),
        )
    return block_fn