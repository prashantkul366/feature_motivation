"""
Transformer block, extracted from TransUNet's vit_seg_modeling.py.

PROVENANCE: `Attention`, `Mlp` and `Block` are lifted from
https://github.com/Beckschen/TransUNet, models/vit_seg_modeling.py -- the same
file the target segmentation model uses. Everything else in that file
(Embeddings, Encoder, DecoderCup, SegmentationHead, VisionTransformer,
load_from, ResNetV2) is TransUNet plumbing we do not need.

LOCAL CHANGES, all forced by the controlled-comparison requirement:

  1. Config object replaced with plain kwargs (no ml_collections dependency).
  2. `Block.forward` returned (x, attn_weights); the wrapper returns x only so
     it drops straight into nn.Sequential.
  3. dropout_rate and attention_dropout_rate are FORCED to 0.0. ViT-B_16
     defaults to 0.1. KAN and VSS carry no dropout, so any nonzero value here
     is a regulariser present in one model and absent from the other two --
     a straight confound in the similarity numbers.
  4. Mlp._init_weights (xavier_uniform + bias std=1e-6) is dropped in favour
     of the single shared init in stem.py, so all three models initialise
     their generic Linears identically.

Scale: 768 -> 192 dims, 12 -> 3 heads, 12 -> 3 layers. The MECHANISM is
unchanged; only width and depth are reduced. Disclose in the paper.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class Attention(nn.Module):
    """Multi-head self-attention. Structure preserved from TransUNet."""

    def __init__(self, hidden_size=192, num_heads=3, attn_dropout=0.0):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} not divisible by "
                             f"num_heads {num_heads}")
        self.num_attention_heads = num_heads
        self.attention_head_size = hidden_size // num_heads
        self.all_head_size = num_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.out = nn.Linear(hidden_size, hidden_size)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads,
                                     self.attention_head_size)
        return x.view(*new_shape).permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        q = self.transpose_for_scores(self.query(hidden_states))
        k = self.transpose_for_scores(self.key(hidden_states))
        v = self.transpose_for_scores(self.value(hidden_states))

        scores = torch.matmul(q, k.transpose(-1, -2))
        scores = scores / math.sqrt(self.attention_head_size)
        probs = self.attn_dropout(self.softmax(scores))

        ctx = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous()
        ctx = ctx.view(*ctx.size()[:-2], self.all_head_size)
        return self.proj_dropout(self.out(ctx))


class Mlp(nn.Module):
    """Feed-forward network. TransUNet's per-module init deliberately removed."""

    def __init__(self, hidden_size=192, mlp_dim=768, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.act_fn = nn.functional.gelu
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.act_fn(self.fc1(x)))
        return self.dropout(self.fc2(x))


class ViTBlock(nn.Module):
    """Pre-norm Transformer block: x + Attn(LN(x)); x + MLP(LN(x))."""

    def __init__(self, hidden_size=192, num_heads=3, mlp_dim=768,
                 dropout_rate=0.0, attention_dropout_rate=0.0,
                 layer_norm_eps=1e-6):
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attn = Attention(hidden_size, num_heads, attention_dropout_rate)
        self.ffn = Mlp(hidden_size, mlp_dim, dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, D) -> (B, N, D)."""
        x = x + self.attn(self.attention_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


def make_vit_block(cfg: dict):
    """Returns a block_fn(index) -> ViTBlock for MotivationNet."""
    def block_fn(_idx: int) -> ViTBlock:
        return ViTBlock(
            hidden_size=cfg["embed_dim"],
            num_heads=cfg["num_heads"],
            mlp_dim=cfg["mlp_dim"],
            dropout_rate=cfg["dropout_rate"],
            attention_dropout_rate=cfg["attention_dropout_rate"],
            layer_norm_eps=cfg["layer_norm_eps"],
        )
    return block_fn