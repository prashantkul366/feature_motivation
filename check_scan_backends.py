"""
Check every available selective-scan backend for NUMERICAL EQUIVALENCE and speed.

    python check_scan_backends.py

WHY THIS EXISTS
Three implementations of the same recurrence may be available:

  torch     selective_scan_torch -- the reference. A Python loop over L
            timesteps. Correct by definition, but kernel-launch-bound:
            measured 172 img/s fwd+bwd for a 3-block VSS model on an
            A100-80GB, i.e. ~22 GPU-hours for three seeds.

  parallel  selective_scan_parallel -- added in this repo. Same mathematics
            evaluated as a parallel prefix scan in ceil(log2(L)) rounds
            instead of L sequential steps, exploiting the associativity of
            x -> a*x + b composition.

  cuda      mamba_ssm's selective_scan_cuda, auto-detected by csms6s.py if
            importable.

THE TRAP THIS SCRIPT GUARDS AGAINST
csms6s.py selects the backend AUTOMATICALLY:

    fn = selective_scan_torch if backend == "torch" or (not WITH_CUDA)
         else SelectiveScanCuda.apply

So merely pip-installing mamba_ssm into the session flips the VSS arm from the
torch reference to a CUDA kernel WITHOUT any change to this repo. That is fine
if and only if the two agree numerically. If they disagree, the VSS arm would
be measuring a different computation from the one this study was designed
around, and nothing would look wrong. Hence: verify first, then choose, then
record the choice in the paper.

Equivalence is checked on FORWARD and on GRADIENTS, at fp32, on shapes
matching the real model (K=4 scan directions, d_inner=192, d_state=16, L=64).
"""

from __future__ import annotations

import sys
import time

import torch

from models.vmamba.csms6s import (
    selective_scan_torch, selective_scan_parallel,
    WITH_CUDA, WITH_SELECTIVESCAN_OFLEX, WITH_SELECTIVESCAN_CORE,
    WITH_SELECTIVESCAN_MAMBA,
)

# Shapes matching the real model: batch 8 for a quick check, d_inner 192,
# K=4 cross-scan directions so KCdim = 768, d_state 16, L = 8*8 grid.
BATCH, K, CDIM, N, L = 8, 4, 192, 16, 64
KCDIM = K * CDIM
# RELATIVE tolerances. Absolute ones are meaningless here: gradient
# magnitudes span |d/du| ~ 5.7e1 to |d/dA| ~ 8.1e3, so a single absolute
# threshold either passes everything or fails d/dA spuriously. fp32 epsilon
# is 1.2e-7; measured agreement between the parallel scan and the reference
# is 2-4e-7 relative and does NOT grow with L (1.7e-7 at L=8, 3.6e-7 at
# L=64), which is what distinguishes float reordering from a logic bug.
TOL_FWD_REL, TOL_GRAD_REL = 1e-4, 1e-4


def make_inputs(device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    mk = lambda *s: torch.randn(*s, generator=g).to(device)
    u = mk(BATCH, KCDIM, L)
    delta = 0.5 * torch.rand(BATCH, KCDIM, L, generator=g).to(device)
    A = -0.5 * torch.rand(KCDIM, N, generator=g).to(device)
    B = mk(BATCH, K, N, L)
    C = mk(BATCH, K, N, L)
    D = mk(KCDIM)
    delta_bias = 0.5 * torch.rand(KCDIM, generator=g).to(device)
    return [t.requires_grad_(True) for t in (u, delta, A, B, C, D, delta_bias)]


def run(fn, device, seed=0, backend=None):
    """Forward + backward, returning output and gradients."""
    u, delta, A, B, C, D, db = make_inputs(device, seed)
    kw = dict(delta_softplus=True, oflex=True)
    if backend is not None:
        out = fn(u, delta, A, B, C, D, db, True, True, backend)
    else:
        out = fn(u, delta, A, B, C, D, db, **kw)
    out.sum().backward()
    grads = [t.grad.clone() for t in (u, delta, A, B, C, D, db)]
    return out.detach().clone(), grads


def rel(x, y):
    """Max relative discrepancy, scaled by the magnitude of the reference."""
    return (x - y).abs().max().item() / max(x.abs().max().item(), 1e-12)


def compare(name_a, a, name_b, b, label):
    oa, ga = a
    ob, gb = b
    rf = rel(oa, ob)
    ok = rf < TOL_FWD_REL
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} forward   "
          f"rel |{name_a} - {name_b}| = {rf:.3e}  (tol {TOL_FWD_REL:.0e})")
    names = ["u", "delta", "A", "B", "C", "D", "delta_bias"]
    worst, worst_n = 0.0, ""
    for n, x, y in zip(names, ga, gb):
        d = rel(x, y)
        if d > worst:
            worst, worst_n = d, n
    gok = worst < TOL_GRAD_REL
    print(f"  [{'PASS' if gok else 'FAIL'}] {label} gradients "
          f"worst rel = {worst:.3e} on d/d{worst_n}  (tol {TOL_GRAD_REL:.0e})")
    return ok and gok


def bench(fn, device, backend=None, iters=20):
    u, delta, A, B, C, D, db = make_inputs(device)
    call = (lambda: fn(u, delta, A, B, C, D, db, True, True, backend)) if backend \
        else (lambda: fn(u, delta, A, B, C, D, db, delta_softplus=True, oflex=True))
    for _ in range(3):
        call().sum().backward()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        for t in (u, delta, A, B, C, D, db):
            t.grad = None
        call().sum().backward()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 74)
    print("SELECTIVE SCAN BACKENDS")
    print("=" * 74)
    print(f"  device: {device}   torch {torch.__version__}")
    print(f"  shapes: batch={BATCH} KCdim={KCDIM} (K={K} x d_inner={CDIM}) "
          f"d_state={N} L={L}")
    print(f"  selective_scan_cuda_oflex : {WITH_SELECTIVESCAN_OFLEX}")
    print(f"  selective_scan_cuda_core  : {WITH_SELECTIVESCAN_CORE}")
    print(f"  selective_scan_cuda       : {WITH_SELECTIVESCAN_MAMBA}")
    print(f"  -> WITH_CUDA = {WITH_CUDA}")
    if WITH_CUDA:
        print("     csms6s will AUTO-SELECT the CUDA kernel unless backend='torch'")
    else:
        print("     csms6s will route to selective_scan_torch")

    failures = []

    print()
    print("=" * 74)
    print("EQUIVALENCE: parallel prefix scan vs the torch reference")
    print("=" * 74)
    ref = run(selective_scan_torch, device)
    par = run(selective_scan_parallel, device)
    if not compare("torch", ref, "parallel", par, "parallel"):
        failures.append("parallel != torch")

    if WITH_CUDA:
        print()
        print("=" * 74)
        print("EQUIVALENCE: CUDA kernel vs the torch reference")
        print("=" * 74)
        if device != "cuda":
            print("  skipped: extensions importable but no CUDA device")
        else:
            try:
                from models.vmamba.csms6s import selective_scan_fn
                cud = run(selective_scan_fn, device, backend=None)
                if not compare("torch", ref, "cuda", cud, "cuda"):
                    failures.append("cuda != torch")
            except Exception as e:
                print(f"  [FAIL] CUDA path raised {type(e).__name__}: {e}")
                print("        The .so was probably built against a different")
                print("        torch ABI. Set VSS['selective_scan_backend']='torch'")
                print("        in config.py and use the parallel scan instead.")
                failures.append("cuda raised")

    print()
    print("=" * 74)
    print("BENCHMARK (fwd + bwd, ms per call, lower is better)")
    print("=" * 74)
    if device != "cuda":
        print("  WARNING: on CPU this benchmark is UNINFORMATIVE. The parallel")
        print("  scan does O(L log L) work versus O(L) for the sequential loop;")
        print("  its only advantage is replacing L sequential KERNEL LAUNCHES")
        print("  with ceil(log2(L)) rounds. With no launch overhead to amortise,")
        print("  the extra work makes it a net loss. Measured on CPU: 1.02x,")
        print("  i.e. nothing. Trust only the CUDA numbers.")
        print()
    t_ref = bench(selective_scan_torch, device)
    t_par = bench(selective_scan_parallel, device)
    print(f"  torch reference   {t_ref:8.2f} ms   1.00x")
    print(f"  parallel scan     {t_par:8.2f} ms   {t_ref/t_par:.2f}x")
    if WITH_CUDA and device == "cuda":
        try:
            from models.vmamba.csms6s import selective_scan_fn
            t_cud = bench(selective_scan_fn, device, backend=None)
            print(f"  cuda kernel       {t_cud:8.2f} ms   {t_ref/t_cud:.2f}x")
        except Exception as e:
            print(f"  cuda kernel       unavailable ({type(e).__name__})")

    print()
    print("=" * 74)
    print("RESULT")
    print("=" * 74)
    if failures:
        print("  FAILED: " + ", ".join(failures))
        print("  Do NOT train VSS with a backend that disagrees with the")
        print("  reference. Pin the working one in config.py.")
        return 1
    print("  All available backends agree with the reference.")
    print("  Pick the fastest and record the choice in the paper.")
    return 0


if __name__ == "__main__":
    sys.exit(main())