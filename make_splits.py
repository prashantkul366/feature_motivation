"""
Generate the frozen index artefacts for the study. Run ONCE, commit the
outputs, never regenerate.

    python make_splits.py

Produces in data/:
    split_indices.npz     train (45000) / val (5000) index arrays over the
                          50000-image CIFAR-100 train set
    analysis_indices.npy  2000 indices into the ORIGINAL train set, drawn
                          from the val portion, used for every feature
                          extraction
    MANIFEST.json         SHA256 of each artefact + the parameters used

WHY THIS IS A SEPARATE, FROZEN STEP
The entire experiment rests on row i of every feature matrix being the same
image for every model and every seed. If the analysis subset were drawn at
extraction time, a differing RNG state between two runs would silently
misalign the rows and every similarity number would be garbage -- and it
would look plausible, not like an error. Freezing to disk and verifying the
hash at load time makes that failure mode impossible rather than unlikely.

The split RNG is seeded with 0 and is INDEPENDENT of the per-run training
seeds (42/123/2024). The data split must not move when the training seed does.
"""

import hashlib
import json
import os

import numpy as np

CIFAR100_TRAIN_SIZE = 50_000
N_VAL = 5_000
N_ANALYSIS = 2_000
N_TOKEN_PAIRS = 2_000
N_TOKENS = 64
SPLIT_SEED = 0

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    rng = np.random.default_rng(SPLIT_SEED)

    perm = rng.permutation(CIFAR100_TRAIN_SIZE)
    val_idx = np.sort(perm[:N_VAL]).astype(np.int64)
    train_idx = np.sort(perm[N_VAL:]).astype(np.int64)

    # Analysis subset is drawn from the VAL portion only: it must never touch
    # training images (representations of memorised images are not what we
    # want to compare) and must never touch the test set (kept sealed).
    analysis_idx = np.sort(
        rng.choice(val_idx, size=N_ANALYSIS, replace=False)).astype(np.int64)

    assert len(train_idx) == CIFAR100_TRAIN_SIZE - N_VAL
    assert len(np.intersect1d(train_idx, val_idx)) == 0, "train/val overlap"
    assert np.isin(analysis_idx, val_idx).all(), "analysis leaked outside val"
    assert len(np.unique(analysis_idx)) == N_ANALYSIS

    # Frozen (image, token) pairs for the token-level comparison. Mean-pooling
    # over 64 tokens collapses spatial arrangement, which is VSS's whole
    # mechanism, so we also compare unpooled features. A full token matrix is
    # 2000*64 = 128k rows and its Gram matrix is not computable; we therefore
    # fix a subsample of exactly N_TOKEN_PAIRS pairs, matching the pooled N so
    # the measured noise floors carry over unchanged.
    tp_rng = np.random.default_rng(SPLIT_SEED)
    pair_img = tp_rng.integers(0, N_ANALYSIS, size=N_TOKEN_PAIRS)
    pair_tok = tp_rng.integers(0, N_TOKENS, size=N_TOKEN_PAIRS)
    token_pairs = np.stack([pair_img, pair_tok], axis=1).astype(np.int64)
    token_pair_path = os.path.join(DATA_DIR, "token_pair_indices.npy")
    np.save(token_pair_path, token_pairs)

    split_path = os.path.join(DATA_DIR, "split_indices.npz")
    analysis_path = os.path.join(DATA_DIR, "analysis_indices.npy")
    np.savez(split_path, train=train_idx, val=val_idx)
    np.save(analysis_path, analysis_idx)

    manifest = {
        "cifar100_train_size": CIFAR100_TRAIN_SIZE,
        "split_seed": SPLIT_SEED,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_analysis": N_ANALYSIS,
        "numpy_generator": "PCG64 via np.random.default_rng",
        "sha256": {
            "split_indices.npz": sha256_of(split_path),
            "analysis_indices.npy": sha256_of(analysis_path),
            "token_pair_indices.npy": sha256_of(token_pair_path),
        },
        "n_token_pairs": N_TOKEN_PAIRS,
        "n_tokens": N_TOKENS,
    }
    with open(os.path.join(DATA_DIR, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"train {len(train_idx)}  val {len(val_idx)}  analysis {len(analysis_idx)}")
    print(f"first 5 analysis indices: {analysis_idx[:5].tolist()}")
    print(f"sha256 analysis_indices.npy: {manifest['sha256']['analysis_indices.npy']}")
    print(f"sha256 split_indices.npz:    {manifest['sha256']['split_indices.npz']}")


def load_splits(verify: bool = True):
    """Load frozen splits, verifying hashes by default.

    Returns:
        (train_idx, val_idx, analysis_idx) int64 arrays.
    """
    # Frozen (image, token) pairs for the token-level comparison. Mean-pooling
    # over 64 tokens collapses spatial arrangement, which is VSS's whole
    # mechanism, so we also compare unpooled features. A full token matrix is
    # 2000*64 = 128k rows and its Gram matrix is not computable; we therefore
    # fix a subsample of exactly N_TOKEN_PAIRS pairs, matching the pooled N so
    # the measured noise floors carry over unchanged.
    tp_rng = np.random.default_rng(SPLIT_SEED)
    pair_img = tp_rng.integers(0, N_ANALYSIS, size=N_TOKEN_PAIRS)
    pair_tok = tp_rng.integers(0, N_TOKENS, size=N_TOKEN_PAIRS)
    token_pairs = np.stack([pair_img, pair_tok], axis=1).astype(np.int64)
    token_pair_path = os.path.join(DATA_DIR, "token_pair_indices.npy")
    np.save(token_pair_path, token_pairs)

    split_path = os.path.join(DATA_DIR, "split_indices.npz")
    analysis_path = os.path.join(DATA_DIR, "analysis_indices.npy")
    manifest_path = os.path.join(DATA_DIR, "MANIFEST.json")

    if verify:
        with open(manifest_path) as f:
            manifest = json.load(f)
        for name, path in (("split_indices.npz", split_path),
                           ("analysis_indices.npy", analysis_path)):
            actual = sha256_of(path)
            expected = manifest["sha256"][name]
            if actual != expected:
                raise RuntimeError(
                    f"{name} hash mismatch -- the frozen split has changed.\n"
                    f"  expected {expected}\n  actual   {actual}\n"
                    "Any features extracted before this change are no longer "
                    "comparable with any extracted after it.")

    z = np.load(split_path)
    return z["train"], z["val"], np.load(analysis_path)


def load_token_pairs(verify: bool = True) -> np.ndarray:
    """Frozen (image_row, token_index) pairs, shape [N_TOKEN_PAIRS, 2]."""
    path = os.path.join(DATA_DIR, "token_pair_indices.npy")
    if verify:
        with open(os.path.join(DATA_DIR, "MANIFEST.json")) as f:
            expected = json.load(f)["sha256"]["token_pair_indices.npy"]
        if sha256_of(path) != expected:
            raise RuntimeError("token_pair_indices.npy hash mismatch")
    return np.load(path)


if __name__ == "__main__":
    main()