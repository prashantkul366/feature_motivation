# Running this on Colab A100

## What this repo is

A motivation study for the KAN + Mamba(VSS) + Transformer segmentation paper.
Three CIFAR-100 classifiers, byte-identical except for the block type. Freeze
them, push the same 2000 images through each, measure how similar the
representations are.

Supports exactly one claim:

> Under matched capacity and identical training, KAN, VSS and Transformer
> blocks converge to representations measurably less similar to each other
> than the same block is to itself across random seeds, at comparable
> CIFAR-100 accuracy.

Does not support: that the differences are useful, that fusion helps, or that
any of this transfers to segmentation. Those are the ablation section's job.

---

## Zero pip installs

Everything upstream that needed compiling has been removed:

| Removed | Why it's fine |
|---|---|
| `mamba-ssm`, `causal-conv1d` | `csms6s.selective_scan_torch` is the reference implementation. At 64 timesteps the speed loss is tolerable. |
| `triton` kernels | The 8x8 grid uses 64 of 1024 lanes in a 32x32-tiled kernel. Pure torch is faster here. |
| `fvcore` | Only fed `VSSM.flops()`, which needs a CUDA op registration that does not exist on the torch path. Report params + img/s instead. |
| `timm` | `drop_path=0` is identity; `trunc_normal_` is in stock PyTorch. |
| `mamba2/ssd_minimal` | Only used by `forward_type="m0"`. Upstream's bare try/except re-raises and made the whole module unimportable without it. |
| `ml_collections` | ViT config is plain kwargs. |

Stock Colab has torch, numpy, scipy, sklearn, torchvision. That is the whole
dependency list.

---

## Setup

```python
# Cell 1 -- mount Drive for checkpoints that survive session death
from google.colab import drive
drive.mount('/content/drive')

!mkdir -p /content/drive/MyDrive/kmt-motivation/{checkpoints,feats,results}
```

```python
# Cell 2 -- get the code
%cd /content
!git clone https://github.com/<you>/kmt-motivation.git
%cd kmt-motivation

# symlink outputs to Drive so a dead session costs nothing
!rm -rf checkpoints feats results
!ln -s /content/drive/MyDrive/kmt-motivation/checkpoints checkpoints
!ln -s /content/drive/MyDrive/kmt-motivation/feats feats
!ln -s /content/drive/MyDrive/kmt-motivation/results results
```

```python
# Cell 3 -- confirm you actually got an A100
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

```python
# Cell 4 -- PREFLIGHT. Never skip this.
!python preflight.py
```

Preflight takes ~30s and checks what would otherwise surface three hours into
a run: all three models build, the parity gate passes, gradients reach every
parameter, the selective scan resolved to a backend, the frozen splits hash
correctly, the stem is identical across architectures, and KAN's grid coverage
is adequate. It exits nonzero on failure, so it gates the cell.

Expected output includes:

```
vit          444,864   1,375,972         +0.95%
kan          410,368   1,272,484         -6.64%
vss          466,368   1,440,484         +5.69%
[PASS] all within +/-15% of mean: worst deviation 6.64%
```

```python
# Cell 5 -- validate the metrics before trusting any number they produce
!python analysis/test_similarity.py
```

Expect `19 passed, 0 failed`. This also prints the empirical noise floors
(SVCCA 0.135, biased RBF CKA 0.141 at N=2000) that the real results must be
read against.

---

## Session strategy

Nine runs, roughly 4.7 GPU-hours in fp32. Colab will disconnect before you
finish. Plan for it rather than fighting it.

**Order runs by architecture, not by seed.** Finish all three ViT seeds first
(fastest, ~45 min total), so you have a complete architecture banked early. If
Colab cuts you off mid-VSS you still have something.

| Session | Runs | Approx |
|---|---|---|
| 1 | preflight + tests + vit x3 | ~1.0 h |
| 2 | kan x3 | ~1.7 h |
| 3 | vss x3 | ~2.2 h |
| 4 | extract + analyse | ~15 min |

Every run writes `checkpoints/{arch}_seed{seed}.pt` to Drive and skips if that
file already exists, so re-running a cell after a disconnect resumes rather
than restarts.

**Keep the tab foregrounded.** Colab reclaims idle sessions. Do not rely on
JS keep-alive hacks; they get you rate-limited.

---

## If you get an L4 or T4 instead of an A100

`selective_scan_torch` materialises `deltaA`, `deltaB_u` and expanded `B`/`C`,
each `(batch, K*d_inner, L, N)`. At batch 128 that is ~7-9 GB across three
blocks. Fine on a 40 GB A100, OOM on a 22 GB L4 or 16 GB T4.

If you OOM:

```python
# in config.py
TRAIN["batch_size"] = 64
TRAIN["lr"] = 7e-4
```

**Apply this to all three models, not just VSS.** Changing the batch size for
one architecture and not the others breaks the controlled comparison outright
and there is no way to repair it afterwards.

Memory escape hatch if still squeezed: `VSS["ssm_d_state"] = 1` cuts those
tensors 16x and is what VMamba's own tuned `s1l8`/`s2l5` configs use. Params
drop 466K -> 432K, still inside parity. Prefer 16 for a more representative
SSM, but it is there.

---

## What to expect, and what would mean something is wrong

**Accuracy: 40-55% top-1.** Not 87%. A 3-block model at dim 192 on CIFAR-100
lands in the low fifties at best. The source paper's 87.8% is not CIFAR-100
top-1 at this depth -- do not calibrate against it, and do not panic when ViT
reports 51%.

**KAN will probably be the weakest.** Budget for it. The pre-declared gates:

| Max pairwise accuracy gap | Action |
|---|---|
| <= 3 pp | Clean. Write "comparable performance." |
| 3-6 pp | Report the gap explicitly, soften the wording. |
| > 6 pp | ONE pre-declared LR sweep `{5e-4, 1e-3, 2e-3}` applied to ALL THREE models, best-val-picked, disclosed. If the gap survives, drop the parity clause rather than tuning until it fits. |

**Re-check KAN grid coverage after epoch 1.** Preflight shows ~0.955 at init,
but activation statistics shift during training. If it falls below 0.90, the
spline branch is going inactive and KANLinear is degrading toward SiLU+Linear
-- at which point you are comparing an MLP to a Transformer and calling it
KAN. Widen `grid_range` if so.

**Kill criterion, committed in advance.** If any cross-architecture SVCCA
falls within 1 std of the corresponding within-architecture (seed control)
value, the architectural difference is not distinguishable from initialisation
noise and the motivation claim must be dropped or heavily qualified. Writing
this down now is what stops the analysis drifting into confirmation-seeking
later.

---

## Reproducibility caveat, state it in the paper

KAN's `spline_weight` is initialised through `torch.linalg.lstsq`, whose
LAPACK driver is not bitwise deterministic -- reduction order varies across
threads. Measured discrepancy is ~1e-8 on values of order 1e-3. Numerically
irrelevant, but it means exact bitwise re-runs are not guaranteed. This is one
reason every number is reported as mean +/- std over three seeds rather than
as a point value. (Full bitwise determinism is unachievable on GPU anyway.)

Do not set `torch.use_deterministic_algorithms(True)` -- it throws on the
scatter/index ops inside the selective scan and buys nothing here.

---

## Repo layout

```
kmt-motivation/
├── config.py                 all hyperparameters + pre-declared gates
├── make_splits.py            run ONCE, commit the output, never regenerate
├── preflight.py              run FIRST on Colab, gates everything
├── data/
│   ├── split_indices.npz     45k/5k, seed 0
│   ├── analysis_indices.npy  the 2000 images, hash-verified on load
│   └── MANIFEST.json         SHA256 of both
├── models/
│   ├── stem.py               shared stem + wrapper, identical for all three
│   ├── block_vit.py          from TransUNet vit_seg_modeling.py
│   ├── block_kan.py          KAN-Mixer (token mix + channel mix)
│   ├── block_vss.py          token <-> 8x8 grid adapter
│   ├── efficient_kan.py      vendored verbatim, do not edit
│   └── vmamba/
│       ├── vss.py            SS2D v2 path + VSSBlock, trimmed
│       ├── csms6s.py         selective scan, torch fallback
│       └── csm_triton.py     cross scan/merge, torch only
└── analysis/
    ├── cca_core.py           Google SVCCA, 2 upstream bugs patched
    ├── cka_core.py           Kornblith CKA + raw HSIC
    ├── similarity.py         protocol v1.0 -- the only judgment calls
    └── test_similarity.py    19 checks, run before trusting any number
```

The `data/` files are frozen artefacts. `analysis_indices.npy` hashes to
`b5a08512...`; `load_splits()` verifies this on every call, so a changed split
fails loudly instead of silently misaligning feature rows -- which would
produce plausible-looking garbage rather than an error.

---

## The run commands

```python
# Cell 6 -- smoke test the loop before committing GPU hours (2 epochs, fake data)
!python train.py --arch vit --seed 42 --smoke
```

```python
# Cell 7 -- cache CIFAR-100 on Drive so it survives session death
!mkdir -p /content/drive/MyDrive/kmt-motivation/cifar_data
!ln -sfn /content/drive/MyDrive/kmt-motivation/cifar_data cifar_data
```

```python
# Cell 8 -- SESSION 1: all three ViT seeds (fastest; bank a whole architecture)
for seed in [42, 123, 2024]:
    !python train.py --arch vit --seed {seed}
```

```python
# Cell 9 -- SESSION 2
for seed in [42, 123, 2024]:
    !python train.py --arch kan --seed {seed}
```

```python
# Cell 10 -- SESSION 3
for seed in [42, 123, 2024]:
    !python train.py --arch vss --seed {seed}
```

```python
# Cell 11 -- SESSION 4: extract and analyse
!python extract_features.py --all
!python run_analysis.py
```

Runs skip if `checkpoints/{arch}_seed{seed}.pt` exists, and resume mid-run from
`{arch}_seed{seed}_last.pt`, so re-running a cell after a disconnect costs
nothing. Same for extraction against `feats/`.

### Revised timing at 100 epochs

| Session | Runs | Approx |
|---|---|---|
| 1 | preflight + tests + vit x3 | ~0.8 h |
| 2 | kan x3 | ~1.2 h |
| 3 | vss x3 | ~1.5 h |
| 4 | extract + analyse | ~15 min |

~3.5 GPU-hours total. Two sessions if Colab is generous.

---

## Reading the output

`run_analysis.py` prints two tables, a diagnostics block, and a verdict.

**Table 1** is the sanity check. What you need is comparable accuracy, not high
accuracy — the argument is "despite similar task performance, the internal
representations differ". The gate verdict prints automatically.

**Table 2** is the result. Every metric is a SIMILARITY: lower means more
diverse. The rows that matter are the `(seed ctrl)` ones — they say what
"similar" looks like when the mechanism is identical and only the seed changed.
A cross-architecture SVCCA of 0.39 means nothing on its own; 0.39 against a
same-architecture baseline of 0.71 is a result.

**PCA-50 truncation diagnostics.** Check these before believing any SVCCA
number. If top-50 variance is low, part of the measured dissimilarity is
truncation artefact rather than architecture. If effective rank is well below
50, the trailing components are noise that cannot align across models and will
drag SVCCA down for reasons unrelated to the block.

**Kill criterion.** Prints OK or FAILS per pair. It fires when a cross-arch
SVCCA is not clearly below that pair's own seed controls. It was committed to
before any run, and it is what stops the analysis drifting into
confirmation-seeking.

Also run `python run_analysis.py --view tokens` for the unpooled comparison.
Mean-pooling collapses spatial arrangement, which is VSS's whole mechanism, so
if pooled and token-level disagree that is a finding worth reporting, not a
problem.