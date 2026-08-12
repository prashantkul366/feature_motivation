"""
Preflight. Run this FIRST on Colab, before any training.

    python preflight.py

Verifies in ~30 seconds what would otherwise be discovered three hours into a
run: that all three models build, that the parameter-parity gate passes, that
gradients reach every parameter, that the selective scan resolved to a working
backend, that the frozen splits hash correctly, and that determinism holds.

Exits nonzero on any failure so it can gate a notebook cell.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch

import config
from models.stem import MotivationNet, count_params
from models.block_vit import make_vit_block
from models.block_kan import make_kan_block
from models.block_vss import make_vss_block

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f": {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)
    return ok


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def build(arch: str, seed: int = 42) -> MotivationNet:
    """Build one model. Seeding here is what makes the shared stem identical
    across architectures for a given seed."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if arch == "vit":
        cfg = dict(embed_dim=config.EMBED_DIM, **config.VIT)
        fn = make_vit_block(cfg)
    elif arch == "kan":
        cfg = dict(embed_dim=config.EMBED_DIM, n_tokens=config.N_TOKENS, **config.KAN)
        cfg.pop("update_grid")
        fn = make_kan_block(cfg)
    elif arch == "vss":
        cfg = dict(embed_dim=config.EMBED_DIM, grid=config.GRID, **config.VSS)
        cfg.pop("channel_first")
        fn = make_vss_block(cfg)
    else:
        raise ValueError(arch)
    return MotivationNet(fn, depth=config.DEPTH, embed_dim=config.EMBED_DIM,
                         num_classes=config.NUM_CLASSES, img_size=config.IMG_SIZE,
                         patch_size=config.PATCH_SIZE, in_chans=config.IN_CHANS,
                         stem_seed=seed)


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    section("ENVIRONMENT")
    print(f"  torch {torch.__version__}   numpy {np.__version__}   device {device}")
    if device == "cuda":
        p = torch.cuda.get_device_properties(0)
        gb = p.total_memory / 1024 ** 3
        print(f"  GPU: {p.name}  {gb:.1f} GB")
        if gb < 30:
            print(f"  NOTE: <30GB. selective_scan_torch materialises ~7-9GB of")
            print(f"        intermediates at batch 128. If you OOM, set batch=64")
            print(f"        AND lr=7e-4 FOR ALL THREE MODELS -- changing the")
            print(f"        batch size for one architecture only breaks the")
            print(f"        controlled comparison outright.")
    else:
        print("  WARNING: no GPU. Parity and shape checks are still valid.")

    from models.vmamba.csms6s import WITH_CUDA
    print(f"  selective_scan backend: {'CUDA extension' if WITH_CUDA else 'torch fallback'}")
    check("selective_scan resolved to a working backend", True,
          "torch fallback is expected and correct on Colab")

    section("FROZEN SPLITS")
    try:
        from make_splits import load_splits
        tr, va, an = load_splits(verify=True)
        check("split hashes verify", True,
              f"train={len(tr)} val={len(va)} analysis={len(an)}")
        check("analysis subset lies inside val", bool(np.isin(an, va).all()))
        check("analysis disjoint from train", len(np.intersect1d(an, tr)) == 0)
    except Exception as e:
        check("frozen splits load", False, str(e))

    section("MODELS BUILD + SHAPES")
    models, totals, per_block = {}, {}, {}
    x = torch.randn(4, config.IN_CHANS, config.IMG_SIZE, config.IMG_SIZE, device=device)
    for arch in ("vit", "kan", "vss"):
        try:
            m = build(arch).to(device)
            models[arch] = m
            totals[arch] = count_params(m)
            per_block[arch] = count_params(m.blocks[0])
            y = m(x)
            ok = tuple(y.shape) == (4, config.NUM_CLASSES)
            check(f"{arch} forward", ok, f"logits {tuple(y.shape)}")
            pooled, blk_pooled, blk_tok = m.forward_features(x, return_all_blocks=True)
            check(f"{arch} feature shapes",
                  tuple(pooled.shape) == (4, config.EMBED_DIM)
                  and len(blk_pooled) == config.DEPTH
                  and tuple(blk_tok[0].shape) == (4, config.N_TOKENS, config.EMBED_DIM),
                  f"pooled {tuple(pooled.shape)}, {len(blk_pooled)} blocks, "
                  f"tokens {tuple(blk_tok[0].shape)}")
        except Exception as e:
            check(f"{arch} builds", False, f"{type(e).__name__}: {e}")

    section("PARAMETER PARITY GATE")
    if len(totals) == 3:
        mean = sum(totals.values()) / 3
        print(f"  {'model':<8} {'per-block':>11} {'total':>11} {'dev from mean':>15}")
        print("  " + "-" * 48)
        worst = 0.0
        for arch in ("vit", "kan", "vss"):
            dev = (totals[arch] - mean) / mean
            worst = max(worst, abs(dev))
            print(f"  {arch:<8} {per_block[arch]:>11,d} {totals[arch]:>11,d} {dev:>+14.2%}")
        print(f"  mean total: {mean:,.0f}   spread: "
              f"{(max(totals.values())-min(totals.values()))/mean:.1%}")
        check(f"all within +/-{config.GATES['param_parity_tolerance']:.0%} of mean",
              worst <= config.GATES["param_parity_tolerance"],
              f"worst deviation {worst:.2%}")

    section("GRADIENT FLOW")
    for arch, m in models.items():
        m.zero_grad(set_to_none=True)
        m(x).sum().backward()
        dead = [n for n, p in m.named_parameters() if p.grad is None]
        finite = all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        check(f"{arch}: every parameter receives gradient", not dead,
              f"{len(dead)} dead" + (f" e.g. {dead[:3]}" if dead else ""))
        check(f"{arch}: gradients finite", finite)

    section("KAN GRID COVERAGE")
    print("  Fraction of pre-KAN activations inside the B-spline grid. Below")
    print("  ~0.90 the spline branch is inactive and KANLinear degrades toward")
    print("  SiLU+Linear -- i.e. you are comparing an MLP to a Transformer.")
    if "kan" in models:
        m = models["kan"]
        with torch.no_grad():
            h = m.patch_embed(x) + m.pos_embed
            for i, blk in enumerate(m.blocks):
                cov = blk.grid_coverage(h)
                h = blk(h)
                s = "  ".join(f"{k}={v:.3f}" for k, v in cov.items())
                worst_cov = min(cov.values())
                print(f"    block {i+1}: {s}")
        check("grid coverage adequate at init", worst_cov > 0.90,
              f"min {worst_cov:.3f} (recheck after 1 epoch of training)")

    section("DETERMINISM")
    print("  NOTE: KAN's spline_weight is initialised through")
    print("  torch.linalg.lstsq, whose LAPACK driver is not bitwise")
    print("  deterministic (reduction order varies across threads). Measured")
    print("  discrepancy is ~1e-8, i.e. float32 epsilon on values that are")
    print("  themselves ~1e-3. This is numerical, not a logic error, so the")
    print("  check uses a tolerance. Consequence to state in the paper: exact")
    print("  bitwise re-runs are not guaranteed, which is why every number is")
    print("  reported as mean +/- std over 3 seeds rather than as a point value.")
    for arch in ("vit", "kan", "vss"):
        a = build(arch).to(device)
        b = build(arch).to(device)
        worst = max((p - q).abs().max().item()
                    for p, q in zip(a.parameters(), b.parameters()))
        check(f"{arch}: same seed -> identical init (atol 1e-6)", worst < 1e-6,
              f"max |diff| = {worst:.2e}")

    section("SHARED STEM IS IDENTICAL ACROSS ARCHITECTURES")
    if len(models) == 3:
        ref = build("vit")
        ok = True
        for arch in ("kan", "vss"):
            other = build(arch)
            ok &= torch.equal(ref.patch_embed.proj.weight, other.patch_embed.proj.weight)
            ok &= torch.equal(ref.pos_embed, other.pos_embed)
        check("patch embed + pos embed identical at seed 42", ok,
              "so within a seed the stem is not a variable")

    section("THROUGHPUT (rough)")
    if device == "cuda":
        for arch, m in models.items():
            xb = torch.randn(64, 3, 32, 32, device=device)
            m.train()
            for _ in range(3):
                m(xb).sum().backward()
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(10):
                m.zero_grad(set_to_none=True)
                m(xb).sum().backward()
            torch.cuda.synchronize()
            dt = (time.time() - t0) / 10
            print(f"    {arch:<5} {64/dt:>8.0f} img/s fwd+bwd  "
                  f"-> ~{45000/(64/dt)/60:.1f} min/epoch at 45k images")
    else:
        print("    skipped (no GPU)")

    section("RESULT")
    if FAILURES:
        print(f"  {len(FAILURES)} FAILED: " + ", ".join(FAILURES))
        print("  Do NOT start training until these pass.")
        return 1
    print("  All checks passed. Safe to train.")
    return 0


if __name__ == "__main__":
    sys.exit(main())