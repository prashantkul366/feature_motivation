"""
Extract representations from a trained checkpoint.

    python extract_features.py --arch vit --seed 42
    python extract_features.py --all          # every checkpoint found

Writes feats/{arch}_seed{seed}.npz containing, for each of the 3 blocks:
    pooled_b{k}   [2000, 192]   mean-pooled over the 64 tokens
    tokens_b{k}   [2000, 192]   unpooled, at 2000 frozen (image, token) pairs

THE ONE INVARIANT THAT MATTERS
Row i of every matrix must be the same image for every model and every seed.
That is guaranteed by data/analysis_indices.npy, whose SHA256 is checked on
load, plus eval-mode, no augmentation, shuffle=False, and a sorted index
array. If this invariant broke, the similarity numbers would not error --
they would come out looking plausible and be meaningless. Hence the
row-alignment guard in analysis/test_similarity.py CASE 9.

WHY TOKEN-LEVEL FEATURES TOO
Mean-pooling over 64 tokens collapses spatial arrangement, and directional
scanning over 2D space is VSS's entire mechanism. Pooling may therefore wash
out exactly the difference we are trying to measure. The token-level view is
cheap insurance: if pooled and token-level disagree, that is a finding worth
reporting, not a problem.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch

import config
from make_splits import load_splits, load_token_pairs
from train import build_model, set_seed

ROOT = os.path.dirname(os.path.abspath(__file__))


def build_analysis_loader(data_root: str, analysis_idx: np.ndarray, smoke=False):
    """Deterministic, unaugmented loader over exactly the frozen indices."""
    if smoke:
        from torch.utils.data import TensorDataset, DataLoader
        g = torch.Generator().manual_seed(0)
        n = len(analysis_idx)
        ds = TensorDataset(torch.randn(n, 3, 32, 32, generator=g),
                           torch.zeros(n, dtype=torch.long))
        return DataLoader(ds, batch_size=256, shuffle=False)

    import torchvision.transforms as T
    from torchvision.datasets import CIFAR100
    from torch.utils.data import DataLoader, Subset

    tf = T.Compose([T.ToTensor(),
                    T.Normalize(config.CIFAR100_MEAN, config.CIFAR100_STD)])
    full = CIFAR100(data_root, train=True, download=True, transform=tf)
    # analysis_idx is sorted, and shuffle=False, so batch order is fixed and
    # row i corresponds to analysis_idx[i] for every model.
    return DataLoader(Subset(full, analysis_idx), batch_size=256,
                      shuffle=False, num_workers=2, pin_memory=True)


@torch.no_grad()
def extract(model, loader, device, token_pairs: np.ndarray) -> dict:
    model.eval()
    pooled = {k: [] for k in range(config.DEPTH)}
    tokens = {k: [] for k in range(config.DEPTH)}
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        _, blk_pooled, blk_tokens = model.forward_features(x, return_all_blocks=True)
        for k in range(config.DEPTH):
            pooled[k].append(blk_pooled[k].float().cpu().numpy())
            tokens[k].append(blk_tokens[k].float().cpu().numpy())

    out = {}
    img_i, tok_i = token_pairs[:, 0], token_pairs[:, 1]
    for k in range(config.DEPTH):
        p = np.concatenate(pooled[k], axis=0)
        t = np.concatenate(tokens[k], axis=0)          # [N, 64, 192]
        out[f"pooled_b{k+1}"] = p.astype(np.float32)
        out[f"tokens_b{k+1}"] = t[img_i, tok_i].astype(np.float32)
    return out


def run_one(arch: str, seed: int, ckpt_dir: str, feat_dir: str,
            data_root: str, device: str, smoke: bool = False) -> bool:
    ckpt_path = os.path.join(ckpt_dir, f"{arch}_seed{seed}.pt")
    out_path = os.path.join(feat_dir, f"{arch}_seed{seed}.npz")
    if not os.path.exists(ckpt_path):
        print(f"[miss] no checkpoint {ckpt_path}")
        return False
    if os.path.exists(out_path):
        print(f"[skip] {out_path} exists")
        return True

    _, _, analysis_idx = load_splits(verify=True)
    token_pairs = load_token_pairs(verify=True)

    set_seed(seed)
    model = build_model(arch, seed).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])

    loader = build_analysis_loader(data_root, analysis_idx, smoke)
    feats = extract(model, loader, device, token_pairs)

    meta = dict(arch=arch, seed=seed, val_top1=ck.get("val_top1", -1.0),
                val_top5=ck.get("val_top5", -1.0), epoch=ck.get("epoch", -1),
                n_params=ck.get("n_params", -1),
                analysis_n=len(analysis_idx))
    np.savez_compressed(out_path, **feats,
                        meta=np.array([repr(meta)], dtype=object))
    shapes = {k: v.shape for k, v in feats.items()}
    print(f"[ok] {out_path}  top1={meta['val_top1']:.4f}  "
          f"pooled_b3={shapes['pooled_b3']}  tokens_b3={shapes['tokens_b3']}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["vit", "kan", "vss"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--ckpt-dir", default=os.path.join(ROOT, "checkpoints"))
    ap.add_argument("--feat-dir", default=os.path.join(ROOT, "feats"))
    ap.add_argument("--data-root", default=os.path.join(ROOT, "cifar_data"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.feat_dir, exist_ok=True)

    if args.all:
        jobs = []
        for p in sorted(glob.glob(os.path.join(args.ckpt_dir, "*_seed*.pt"))):
            base = os.path.basename(p)
            if base.endswith("_last.pt"):
                continue
            arch, rest = base.split("_seed")
            jobs.append((arch, int(rest[:-3])))
        if not jobs:
            print("no checkpoints found")
            return
        print(f"found {len(jobs)} checkpoints")
    else:
        if not (args.arch and args.seed):
            ap.error("give --arch and --seed, or --all")
        jobs = [(args.arch, args.seed)]

    for arch, seed in jobs:
        run_one(arch, seed, args.ckpt_dir, args.feat_dir,
                args.data_root, device, args.smoke)


if __name__ == "__main__":
    main()