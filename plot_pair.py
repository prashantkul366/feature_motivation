"""
Plot the SVCCA curve for ONE architecture pair, without the seed controls.

    python plot_pair.py --a kan --b vit
    python plot_pair.py --a kan --b vit --view tokens --block 2

Writes results/svcca_{a}_{b}_{view}_b{block}.png (and .pdf for LaTeX).

The curve is the mean over all 9 seed pairings (3 x 3), shaded with +/- 1 s.d.

NOTE: without the within-architecture control on the same axes, this figure
shows a descending curve but gives the reader no scale to judge it against --
every SVCCA curve descends. If this panel goes in the paper without the
control, the control numbers need to be adjacent in the text or the table,
or the figure argues nothing on its own.
"""

from __future__ import annotations

import argparse
import ast
import glob
import os

import numpy as np

from analysis.similarity import standardize, svcca

ROOT = os.path.dirname(os.path.abspath(__file__))
PRETTY = {"kan": "KAN", "vss": "Mamba (VSS)", "vit": "Transformer"}


def load_arch(feat_dir: str, arch: str, key: str):
    out = []
    for p in sorted(glob.glob(os.path.join(feat_dir, f"{arch}_seed*.npz"))):
        z = np.load(p, allow_pickle=True)
        meta = ast.literal_eval(str(z["meta"][0]))
        out.append((meta["seed"], z[key]))
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="kan")
    ap.add_argument("--b", default="vit")
    ap.add_argument("--view", default="pooled", choices=["pooled", "tokens"])
    ap.add_argument("--block", type=int, default=3)
    ap.add_argument("--feat-dir", default=os.path.join(ROOT, "feats"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--k", type=int, default=50)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    key = f"{args.view}_b{args.block}"
    A, B = load_arch(args.feat_dir, args.a, key), load_arch(args.feat_dir, args.b, key)
    if not A or not B:
        raise SystemExit(f"missing features for {args.a} or {args.b} in {args.feat_dir}")

    coefs, means = [], []
    for sa, xa in A:
        for sb, xb in B:
            r = svcca(standardize(xa), standardize(xb), n_components=args.k)
            coefs.append(r["coefs"])
            means.append(r["mean"])
    C = np.stack(coefs)
    m, s = C.mean(0), C.std(0, ddof=1)
    label = f"{PRETTY[args.a]}\u2013{PRETTY[args.b]}"
    print(f"{label}: {len(coefs)} seed pairings   "
          f"mean SVCCA = {np.mean(means):.4f} +/- {np.std(means, ddof=1):.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    xs = np.arange(len(m))
    ax.plot(xs, m, color="#1f77b4", lw=2.0, label=label)
    ax.fill_between(xs, m - s, m + s, color="#1f77b4", alpha=0.20,
                    label="$\\pm$1 s.d. over 9 seed pairings")
    ax.set_xlabel("CCA component")
    ax.set_ylabel("Canonical correlation")
    ax.set_title(f"SVCCA correlations: {label}")
    ax.set_xlim(0, len(m) - 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()

    stem = os.path.join(args.out_dir,
                        f"svcca_{args.a}_{args.b}_{args.view}_b{args.block}")
    fig.savefig(stem + ".png", dpi=300)
    fig.savefig(stem + ".pdf")
    print(f"wrote {stem}.png and .pdf")


if __name__ == "__main__":
    main()