"""
Shared stem and model wrapper.

EVERYTHING in this file is byte-identical across the three models. The only
thing that varies between KAN, VSS and Transformer runs is the block passed
into `MotivationNet`. That is what licenses the sentence "the observed
differences are attributable to the architectural mechanism".

    3x32x32
      -> Conv2d(3, 192, k=4, s=4)      IDENTICAL
      -> 64 tokens x 192 dims
      -> + learnable pos embed         IDENTICAL
      -> 3 x {block}                   <-- THE ONLY DIFFERENCE
      -> LayerNorm                     IDENTICAL
      -> mean-pool over tokens         IDENTICAL  <- FEATURE EXTRACTION POINT
      -> Linear(192, 100)              IDENTICAL
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Non-overlapping 4x4 patch embedding: 32x32 -> 8x8 grid of 192-d tokens."""

    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=192):
        super().__init__()
        self.grid = img_size // patch_size
        self.n_tokens = self.grid ** 2
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                 # (B, D, g, g)
        return x.flatten(2).transpose(1, 2)   # (B, N, D)


class MotivationNet(nn.Module):
    """Shared stem + N identical blocks + shared head.

    Args:
        block_fn: callable(index) -> nn.Module. Each block must map
                  (B, N, D) -> (B, N, D). Any internal reshaping (e.g. VSS
                  needing a 2D grid) happens inside the block, so all three
                  architectures see an identical tensor at the boundary.
        depth: number of blocks.
    """

    def __init__(self, block_fn, depth=3, embed_dim=192, num_classes=100,
                 img_size=32, patch_size=4, in_chans=3, norm_eps=1e-6,
                 stem_seed=None):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.n_tokens, embed_dim))
        self.blocks = nn.ModuleList([block_fn(i) for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

        if stem_seed is not None:
            self._reinit_stem(stem_seed)

    def _reinit_stem(self, seed: int) -> None:
        """Re-initialise the shared stem and head from a DEDICATED RNG stream.

        WHY THIS IS NECESSARY, NOT DEFENSIVE PROGRAMMING
        Block construction consumes RNG, and the three architectures consume
        DIFFERENT AMOUNTS. So any stem parameter initialised after the blocks
        -- which, because apply() runs last, is all of them -- lands at a
        different point in the RNG stream for each architecture. Measured at
        seed 42 before this fix:

            vit  patch_embed[0,0,0,0] = +0.014110   pos_embed[0,0,0] = -0.014400
            kan  patch_embed[0,0,0,0] = +0.028528   pos_embed[0,0,0] = +0.006776
            vss  patch_embed[0,0,0,0] = +0.002728   pos_embed[0,0,0] = -0.025277

        That silently makes the stem a variable in an experiment whose entire
        claim is that the stem is NOT a variable. Drawing stem parameters from
        a separate generator makes them bit-identical across architectures for
        a given seed, independent of what the blocks drew.
        """
        g = torch.Generator().manual_seed(seed)
        nn.init.trunc_normal_(self.pos_embed, std=0.02, generator=g)
        nn.init.trunc_normal_(self.patch_embed.proj.weight, std=0.02, generator=g)
        nn.init.constant_(self.patch_embed.proj.bias, 0)
        nn.init.trunc_normal_(self.head.weight, std=0.02, generator=g)
        nn.init.constant_(self.head.bias, 0)
        nn.init.constant_(self.norm.weight, 1.0)
        nn.init.constant_(self.norm.bias, 0)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """One init rule for all generic Linear/LayerNorm/Conv2d, applied to
        every model.

        Architecture-SPECIFIC inits are deliberately NOT touched here and must
        re-run after this: Mamba's S4D A_log/D/dt_proj init and KAN's spline
        init ARE the mechanism, not incidental choices. Blocks that carry such
        init must restore it in their own __init__ after apply() runs, or be
        excluded via `_skip_generic_init`.
        """
        if getattr(m, "_skip_generic_init", False):
            return
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor, return_all_blocks=False):
        """Returns the 192-d pooled feature. This IS the extraction point.

        Args:
            x: (B, 3, 32, 32)
            return_all_blocks: if True also return per-block pooled features
                               and per-block token-level features.

        Returns:
            pooled (B, D), or (pooled, [per-block pooled], [per-block tokens])
        """
        x = self.patch_embed(x) + self.pos_embed
        per_block_pooled, per_block_tokens = [], []
        for blk in self.blocks:
            x = blk(x)
            if return_all_blocks:
                per_block_tokens.append(x)
                per_block_pooled.append(x.mean(dim=1))
        x = self.norm(x)
        pooled = x.mean(dim=1)
        if return_all_blocks:
            return pooled, per_block_pooled, per_block_tokens
        return pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def count_params(model: nn.Module, trainable_only=True) -> int:
    ps = model.parameters()
    return sum(p.numel() for p in ps if (p.requires_grad or not trainable_only))