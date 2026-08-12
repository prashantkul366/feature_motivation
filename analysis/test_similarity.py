"""
Validation of the representation-similarity pipeline on synthetic data with
known answers. Run before trusting any number produced on real features:

    python analysis/test_similarity.py

Every assertion here has an analytic justification. Cases marked MEASURE do
not assert a target -- they quantify an artefact of the protocol whose size
we need to know in order to read the real results honestly.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.similarity import (
    standardize, svcca, cka_and_hsic, spectrum_diagnostics)
from analysis import cka_core

RNG = np.random.default_rng(0)
N, D = 2000, 192

PASS, FAIL = [], []


def check(name, ok, detail):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def hdr(n, title, note=""):
    print()
    print("=" * 76)
    print(f"CASE {n}: {title}")
    if note:
        for line in note.strip().split("\n"):
            print(f"        {line}")
    print("=" * 76)


def random_orthogonal(d, rng):
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    return q * np.sign(np.diag(r))


X = RNG.standard_normal((N, D))
xs = standardize(X)

# --------------------------------------------------------------------------
hdr(1, "identical representations -> everything exactly 1.0")
s = svcca(xs, xs, n_components=50)
c = cka_and_hsic(xs, xs)
check("svcca(X,X)", abs(s["mean"] - 1.0) < 1e-6, f"{s['mean']:.6f}")
check("cka_linear(X,X)", abs(c["cka_linear"] - 1.0) < 1e-9, f"{c['cka_linear']:.6f}")
check("cka_rbf(X,X)", abs(c["cka_rbf"] - 1.0) < 1e-9, f"{c['cka_rbf']:.6f}")

# --------------------------------------------------------------------------
hdr(2, "CCA invariance to invertible linear maps, tested at FULL rank",
    "Theory: CCA is invariant to any invertible linear transform of the\n"
    "full space. It is NOT invariant after a rank-reducing PCA, because\n"
    "PCA-k of X and PCA-k of XA span different subspaces. So the\n"
    "invariance property must be tested at k = D-1, not at k = 50.")
Q = random_orthogonal(D, RNG)
A = RNG.standard_normal((D, D))
XQ, XA = standardize(X @ Q), standardize(X @ A)
check("svcca(X, XQ) full rank", svcca(xs, XQ, n_components=D - 1)["mean"] > 0.99,
      f"{svcca(xs, XQ, n_components=D-1)['mean']:.6f}")
check("svcca(X, XA) full rank", svcca(xs, XA, n_components=D - 1)["mean"] > 0.99,
      f"{svcca(xs, XA, n_components=D-1)['mean']:.6f}")
check("cka_linear(X, XQ) ~ 1 (orthogonal)", cka_and_hsic(xs, XQ)["cka_linear"] > 0.99,
      f"{cka_and_hsic(xs, XQ)['cka_linear']:.6f}")
check("cka_linear(X, XA) < 0.9 (non-orthogonal)",
      cka_and_hsic(xs, XA)["cka_linear"] < 0.9,
      f"{cka_and_hsic(xs, XA)['cka_linear']:.6f}  <- CKA is NOT invariant here,")
print("        which is the point: SVCCA and CKA measure different things.")

# --------------------------------------------------------------------------
hdr(3, "isotropic scaling -> CKA exactly invariant")
c = cka_and_hsic(xs, standardize(X * 7.3))
check("cka_linear(X, 7.3X)", abs(c["cka_linear"] - 1.0) < 1e-9, f"{c['cka_linear']:.6f}")

# --------------------------------------------------------------------------
hdr(4, "MEASURE: how much does PCA-50 truncation cost SVCCA?",
    "Same underlying representation, reparameterised by an invertible A.\n"
    "Ideal answer is 1.0. Whatever we lose here is pure truncation\n"
    "artefact, and it depends entirely on the spectrum.")
print(f"      {'latent rank':>12} | {'top50 var':>10} | {'eff. rank':>10} | {'svcca(X,XA)':>12}")
print("      " + "-" * 54)
for rank in (192, 150, 100, 50, 20):
    L = RNG.standard_normal((D, rank))
    Z = RNG.standard_normal((N, rank)) @ L.T + 0.3 * RNG.standard_normal((N, D))
    zs = standardize(Z)
    d = spectrum_diagnostics(zs)
    v = svcca(zs, standardize(Z @ A), n_components=50)["mean"]
    print(f"      {rank:>12} | {d['topk_var_frac']:>9.1%} | "
          f"{d['effective_rank']:>10.1f} | {v:>12.4f}")
print()
print("      READ-OFF: SVCCA-50 only approaches its invariance ideal when the")
print("      top-50 subspace holds most of the variance. On real features we")
print("      MUST report top50_var_frac alongside SVCCA, or a low number is")
print("      uninterpretable. Note rank=20: high top-50 variance but low SVCCA,")
print("      because components past the true rank are noise that cannot align.")

# --------------------------------------------------------------------------
hdr(5, "MEASURE + ASSERT: CKA finite-sample floor on INDEPENDENT data",
    "Biased CKA has a positive floor ~ D/N. Debiased removes it.\n"
    "Our operating point is N=2000, D=192.")
print(f"      {'N':>6} | {'bias.lin':>9} {'debias.lin':>11} | {'bias.rbf':>9} {'debias.rbf':>11}")
print("      " + "-" * 54)
floors = {}
for n in (500, 1000, 2000, 5000):
    a = standardize(RNG.standard_normal((n, D)))
    b = standardize(RNG.standard_normal((n, D)))
    gla, glb = cka_core.gram_linear(a), cka_core.gram_linear(b)
    gra, grb = cka_core.gram_rbf(a), cka_core.gram_rbf(b)
    bl, dl = cka_core.cka(gla, glb), cka_core.cka(gla, glb, debiased=True)
    br, dr = cka_core.cka(gra, grb), cka_core.cka(gra, grb, debiased=True)
    floors[n] = (bl, dl, br, dr)
    print(f"      {n:>6} | {bl:>9.4f} {dl:>11.4f} | {br:>9.4f} {dr:>11.4f}")
check("debiased linear CKA ~ 0 at N=2000", abs(floors[2000][1]) < 0.02,
      f"{floors[2000][1]:.4f}")
check("debiased RBF CKA ~ 0 at N=2000", abs(floors[2000][3]) < 0.02,
      f"{floors[2000][3]:.4f}")
check("biased CKA floor is NOT negligible at N=2000", floors[2000][0] > 0.05,
      f"linear {floors[2000][0]:.4f}, rbf {floors[2000][2]:.4f} "
      f"-> report DEBIASED as headline")

# --------------------------------------------------------------------------
hdr(6, "MEASURE: SVCCA noise floor on independent data",
    "Independent gaussians should score 0 in the population limit. They do\n"
    "not at finite N. Real cross-architecture SVCCA must be read against\n"
    "this floor, not against 0.")
print(f"      {'N':>6} | " + " | ".join(f"{'k='+str(k):>7}" for k in (30, 50, 100)))
print("      " + "-" * 36)
sfloor = {}
for n in (500, 1000, 2000, 5000):
    row = []
    for k in (30, 50, 100):
        a = standardize(RNG.standard_normal((n, D)))
        b = standardize(RNG.standard_normal((n, D)))
        v = svcca(a, b, n_components=k)["mean"]
        sfloor[(n, k)] = v
        row.append(f"{v:>7.3f}")
    print(f"      {n:>6} | " + " | ".join(row))
check("SVCCA floor decreases with N (k=50)", sfloor[(5000, 50)] < sfloor[(500, 50)],
      f"N=500:{sfloor[(500,50)]:.3f} -> N=5000:{sfloor[(5000,50)]:.3f}")
print(f"      OPERATING POINT N=2000, k=50 -> SVCCA floor = {sfloor[(2000,50)]:.3f}")

# --------------------------------------------------------------------------
hdr(7, "graded overlap -> all metrics monotone increasing")
prev = {"svcca": -1.0, "cka": -1.0, "dcka": -1.0}
mono = {"svcca": True, "cka": True, "dcka": True}
print(f"      {'shared dims':>12} | {'svcca':>7} | {'cka_lin':>8} | {'debiased':>9}")
print("      " + "-" * 46)
for shared in (0, 48, 96, 144, 192):
    Z = np.concatenate([X[:, :shared], RNG.standard_normal((N, D - shared))], axis=1)
    zs = standardize(Z)
    sv = svcca(xs, zs, n_components=50)["mean"]
    cc = cka_and_hsic(xs, zs)
    mono["svcca"] &= sv > prev["svcca"]
    mono["cka"] &= cc["cka_linear"] > prev["cka"]
    mono["dcka"] &= cc["cka_linear_debiased"] > prev["dcka"]
    prev = {"svcca": sv, "cka": cc["cka_linear"], "dcka": cc["cka_linear_debiased"]}
    print(f"      {str(shared)+'/192':>12} | {sv:>7.4f} | "
          f"{cc['cka_linear']:>8.4f} | {cc['cka_linear_debiased']:>9.4f}")
check("svcca monotone", mono["svcca"], "increasing")
check("biased cka monotone", mono["cka"], "increasing")
check("debiased cka monotone", mono["dcka"], "increasing")

# --------------------------------------------------------------------------
hdr(8, "determinism -- identical inputs must give bit-identical outputs")
a1 = svcca(xs, standardize(RNG.standard_normal((N, D))), n_components=50)["mean"]
b = standardize(RNG.standard_normal((N, D)))
check("svcca deterministic", svcca(xs, b, 50)["mean"] == svcca(xs, b, 50)["mean"],
      f"{svcca(xs, b, 50)['mean']:.12f}")
check("cka deterministic",
      cka_and_hsic(xs, b)["cka_linear"] == cka_and_hsic(xs, b)["cka_linear"],
      f"{cka_and_hsic(xs, b)['cka_linear']:.12f}")

# --------------------------------------------------------------------------
hdr(9, "row-alignment guard -- shuffling one side must destroy similarity",
    "This is the check that catches the single worst silent bug in this\n"
    "kind of study: feature matrices whose row i is not the same image.")
perm = RNG.permutation(N)
sv_ok = svcca(xs, xs, n_components=50)["mean"]
sv_bad = svcca(xs, xs[perm], n_components=50)["mean"]
ck_ok = cka_and_hsic(xs, xs)["cka_linear_debiased"]
ck_bad = cka_and_hsic(xs, xs[perm])["cka_linear_debiased"]
check("shuffle collapses svcca", sv_bad < 0.5 * sv_ok, f"{sv_ok:.4f} -> {sv_bad:.4f}")
check("shuffle collapses debiased cka", abs(ck_bad) < 0.02, f"{ck_ok:.4f} -> {ck_bad:.4f}")

print()
print("=" * 76)
print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
print("=" * 76)
sys.exit(1 if FAIL else 0)