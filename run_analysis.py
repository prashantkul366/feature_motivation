"""
Compute the tables and figure from extracted features.

    python run_analysis.py

Reads feats/*.npz, writes to results/:
    table1_accuracy.csv        per-model accuracy / params (the sanity check)
    table2_similarity.csv      cross-arch and within-arch similarity
    figure1_svcca_curves.png   SVCCA canonical correlations, with the
                               within-architecture control as a reference band
    raw_pairs.json             every individual pairing, before aggregation

THE COMPARISON THAT MAKES THIS A RESULT
Cross-architecture similarity on its own is just a number. What turns it into
evidence is the WITHIN-architecture control: the same block type, different
seed. That measures how much two representations differ when the mechanism is
identical and only initialisation and data order changed. If cross-arch
similarity is not clearly below within-arch, the architectural difference is
indistinguishable from seed noise and the motivation claim fails. That check
is free -- three seeds are being trained anyway -- and it is the difference
between "SVCCA = 0.39" and "SVCCA = 0.39 against a same-architecture baseline
of 0.71".

Cross-arch pairs use all 3x3 = 9 seed combinations; within-arch uses the 3
distinct seed pairs. Reported as mean +/- std over those.
"""

from __future__ import annotations

import argparse
import ast
import glob
import itertools
import json
import os
from collections import defaultdict

import numpy as np

import config
from analysis.similarity import compare, svcca, standardize

ROOT = os.path.dirname(os.path.abspath(__file__))
ARCHS = ("kan", "vss", "vit")
PRETTY = {"kan": "KAN", "vss": "Mamba (VSS)", "vit": "Transformer"}


def load_feats(feat_dir: str) -> dict:
    out = {}
    for p in sorted(glob.glob(os.path.join(feat_dir, "*_seed*.npz"))):
        base = os.path.basename(p)[:-4]
        arch, seed = base.split("_seed")
        z = np.load(p, allow_pickle=True)
        meta = ast.literal_eval(str(z["meta"][0]))
        out[(arch, int(seed))] = dict(
            path=p, meta=meta,
            **{k: z[k] for k in z.files if k != "meta"})
    return out


def aggregate(values: list[float]) -> tuple[float, float]:
    a = np.asarray(values, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", default=os.path.join(ROOT, "feats"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--block", type=int, default=config.DEPTH,
                    help="which block's features to headline (default: last)")
    ap.add_argument("--view", default="pooled", choices=["pooled", "tokens"])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    feats = load_feats(args.feat_dir)
    if not feats:
        print(f"no feature files in {args.feat_dir}")
        return
    key = f"{args.view}_b{args.block}"
    seeds_by_arch = defaultdict(list)
    for (arch, seed) in feats:
        seeds_by_arch[arch].append(seed)
    for a in seeds_by_arch:
        seeds_by_arch[a].sort()
    print(f"loaded {len(feats)} runs: "
          + ", ".join(f"{a}x{len(s)}" for a, s in sorted(seeds_by_arch.items())))
    print(f"analysing {key}\n")

    # ---------------------------------------------------------- Table 1 ----
    print("=" * 78)
    print("TABLE 1  Classification sanity check")
    print("=" * 78)
    rows1, accs = [], {}
    for arch in ARCHS:
        if arch not in seeds_by_arch:
            continue
        t1 = [feats[(arch, s)]["meta"]["val_top1"] for s in seeds_by_arch[arch]]
        t5 = [feats[(arch, s)]["meta"]["val_top5"] for s in seeds_by_arch[arch]]
        npar = feats[(arch, seeds_by_arch[arch][0])]["meta"]["n_params"]
        m1, s1 = aggregate(t1)
        m5, s5 = aggregate(t5)
        accs[arch] = m1
        rows1.append(dict(model=PRETTY[arch], params=npar, n_seeds=len(t1),
                          top1_mean=m1, top1_std=s1, top5_mean=m5, top5_std=s5))
        print(f"  {PRETTY[arch]:<14} params {npar:>10,d}   "
              f"top-1 {m1:6.2%} +/- {s1:.2%}   top-5 {m5:6.2%} +/- {s5:.2%}")

    if len(accs) > 1:
        gap = (max(accs.values()) - min(accs.values())) * 100
        g = config.GATES
        verdict = ("CLEAN -- may write 'comparable performance'"
                   if gap <= g["accuracy_gap_clean"] else
                   "ACCEPTABLE -- report the gap explicitly, soften wording"
                   if gap <= g["accuracy_gap_acceptable"] else
                   "FAILED -- run the pre-declared LR sweep on ALL THREE "
                   "models, or drop the parity clause")
        print(f"\n  max pairwise accuracy gap: {gap:.2f} pp  ->  {verdict}")

    # ---------------------------------------------------------- Table 2 ----
    print()
    print("=" * 78)
    print("TABLE 2  Representation similarity")
    print("=" * 78)

    raw, agg = defaultdict(list), {}
    curves = {}

    def pair_label(a, b):
        return f"{PRETTY[a]}-{PRETTY[b]}"

    # cross-architecture: all seed combinations
    for a, b in itertools.combinations(ARCHS, 2):
        if a not in seeds_by_arch or b not in seeds_by_arch:
            continue
        label, cs = pair_label(a, b), []
        for sa in seeds_by_arch[a]:
            for sb in seeds_by_arch[b]:
                r = compare(feats[(a, sa)][key], feats[(b, sb)][key])
                raw[label].append(dict(seed_a=sa, seed_b=sb,
                                       **{k: v for k, v in r.items()
                                          if not isinstance(v, np.ndarray)}))
                cs.append(r["svcca50_coefs"])
        curves[label] = np.stack(cs)
        agg[label] = ("cross", raw[label])

    # within-architecture control: distinct seed pairs, same block type
    for a in ARCHS:
        if a not in seeds_by_arch or len(seeds_by_arch[a]) < 2:
            continue
        label, cs = f"{PRETTY[a]}-{PRETTY[a]} (seed ctrl)", []
        for sa, sb in itertools.combinations(seeds_by_arch[a], 2):
            r = compare(feats[(a, sa)][key], feats[(a, sb)][key])
            raw[label].append(dict(seed_a=sa, seed_b=sb,
                                   **{k: v for k, v in r.items()
                                      if not isinstance(v, np.ndarray)}))
            cs.append(r["svcca50_coefs"])
        curves[label] = np.stack(cs)
        agg[label] = ("within", raw[label])

    metrics = [("svcca50_mean", "SVCCA"),
               ("cka_linear_debiased_full", "linCKA*"),
               ("cka_rbf_debiased_full", "rbfCKA*"),
               ("hsic_rbf_pca50", "HSIC")]
    print(f"  {'pair':<34}" + "".join(f"{n:>16}" for _, n in metrics))
    print("  " + "-" * 96)
    rows2 = []
    for label, (kind, entries) in agg.items():
        row = dict(pair=label, kind=kind, n=len(entries))
        cells = []
        for mk, _ in metrics:
            m, s = aggregate([e[mk] for e in entries])
            row[f"{mk}_mean"], row[f"{mk}_std"] = m, s
            cells.append(f"{m:>9.4f}+/-{s:.3f}")
        rows2.append(row)
        print(f"  {label:<34}" + "".join(f"{c:>16}" for c in cells))
    print("\n  * debiased CKA. The biased estimator has a floor of "
          f"{config.NOISE_FLOOR['cka_linear_biased']:.3f} (linear) / "
          f"{config.NOISE_FLOOR['cka_rbf_biased']:.3f} (RBF) at N=2000, D=192.")
    print(f"  All metrics are SIMILARITIES: lower = more diverse. Independent "
          f"data scores\n  SVCCA {config.NOISE_FLOOR['svcca_k50']:.3f}, not 0 "
          f"-- read the numbers against that floor.")

    # spectrum diagnostics, required to interpret SVCCA at all
    print()
    print("  PCA-50 truncation diagnostics (SVCCA is uninterpretable without these):")
    print("  If top-50 variance is low, part of the measured dissimilarity is")
    print("  truncation artefact, not architecture. If effective rank is well")
    print("  below 50, trailing components are noise that cannot align.")
    from analysis.similarity import spectrum_diagnostics
    spec_rows = []
    for arch in ARCHS:
        if arch not in seeds_by_arch:
            continue
        fr, er = [], []
        for s in seeds_by_arch[arch]:
            d = spectrum_diagnostics(standardize(feats[(arch, s)][key]))
            fr.append(d["topk_var_frac"])
            er.append(d["effective_rank"])
        mf, sf = aggregate(fr)
        me, se = aggregate(er)
        spec_rows.append(dict(model=PRETTY[arch], top50_var_frac=mf,
                              effective_rank=me))
        print(f"    {PRETTY[arch]:<14} top-50 variance {mf:6.1%} +/- {sf:.1%}"
              f"   effective rank {me:5.1f} +/- {se:.1f}")

    # ------------------------------------------------- KILL CRITERION ------
    print()
    print("=" * 78)
    print("KILL CRITERION")
    print("=" * 78)
    print("  Any cross-architecture SVCCA within 1 std of the corresponding")
    print("  within-architecture control means the architectural difference is")
    print("  not distinguishable from initialisation noise.")
    within = {r["pair"]: r for r in rows2 if r["kind"] == "within"}
    failed = []
    if within:
        def ctrl_for(arch):
            k = f"{PRETTY[arch]}-{PRETTY[arch]} (seed ctrl)"
            return within.get(k)

        for a, b in itertools.combinations(ARCHS, 2):
            label = pair_label(a, b)
            row = next((r for r in rows2 if r["pair"] == label), None)
            ca, cb = ctrl_for(a), ctrl_for(b)
            if row is None or ca is None or cb is None:
                continue
            # compare against THIS pair's own two controls, not a global
            # minimum: a low control for some third architecture says nothing
            # about whether a and b are distinguishable from each other.
            bar = min(ca["svcca50_mean_mean"], cb["svcca50_mean_mean"])
            bar_std = max(ca["svcca50_mean_std"], cb["svcca50_mean_std"],
                          row["svcca50_mean_std"])
            margin = bar - row["svcca50_mean_mean"]
            ok = margin > bar_std
            print(f"    {label:<34} {row['svcca50_mean_mean']:.4f}  vs control "
                  f"{bar:.4f}   margin {margin:+.4f}   {'OK' if ok else 'FAILS'}")
            if not ok:
                failed.append(label)
    print("\n  " + ("VERDICT: cross-architecture similarity is clearly below the "
                    "seed control." if not failed else
                    f"VERDICT: {len(failed)} pair(s) indistinguishable from seed "
                    f"noise. Qualify or drop the claim."))

    # ------------------------------------------------------- artefacts -----
    import csv
    with open(os.path.join(args.out_dir, "table1_accuracy.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows1[0].keys()))
        w.writeheader()
        w.writerows(rows1)
    with open(os.path.join(args.out_dir, "table2_similarity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows2[0].keys()))
        w.writeheader()
        w.writerows(rows2)
    with open(os.path.join(args.out_dir, "raw_pairs.json"), "w") as f:
        json.dump({k: v for k, v in raw.items()}, f, indent=2, default=float)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        greys = ["0.35", "0.55", "0.70"]
        ci = gi = 0
        for label, arr in curves.items():
            m, s = arr.mean(0), arr.std(0)
            xs = np.arange(len(m))
            if "seed ctrl" in label:
                c = greys[gi % len(greys)]
                gi += 1
                ax.plot(xs, m, "--", color=c, lw=1.2, label=label)
                ax.fill_between(xs, m - s, m + s, color=c, alpha=0.15)
            else:
                ax.plot(xs, m, color=colors[ci], lw=1.9, label=label)
                ax.fill_between(xs, m - s, m + s, color=colors[ci], alpha=0.18)
                ci += 1
        ax.set_xlabel("CCA component")
        ax.set_ylabel("Canonical correlation")
        ax.set_title("SVCCA correlations (mean $\\pm$ 1 s.d. over seed pairings)")
        ax.legend(fontsize=8, frameon=False, ncol=2)
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_xlim(0, len(m) - 1)
        ax.set_ylim(0, 1)
        ax.text(0.985, 0.965,
                "dashed = same architecture, different seed\n"
                "(the bar cross-architecture pairs must fall below)",
                transform=ax.transAxes, ha="right", va="top", fontsize=7,
                color="0.35")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "figure1_svcca_curves.png"), dpi=220)
        print(f"\n  wrote figure1_svcca_curves.png")
    except ImportError:
        print("\n  matplotlib unavailable; skipped the figure")

    print(f"  wrote table1_accuracy.csv, table2_similarity.csv, raw_pairs.json")
    print(f"  -> {args.out_dir}")


if __name__ == "__main__":
    main()