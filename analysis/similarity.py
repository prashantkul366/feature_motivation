"""
Representation-similarity protocol for the KAN / Mamba / Transformer
complementarity motivation study.

This is the ONLY file in analysis/ containing our own judgment calls.
cca_core.py and cka_core.py are vendored reference implementations and must
not be edited to make numbers come out differently.

PROTOCOL v1.0 (frozen -- changing any of this invalidates comparability
with results already produced):

  1. Features arrive as [N, D] float arrays, N examples x D dims, where row i
     is THE SAME IMAGE for every model. This is enforced upstream by a shared
     analysis_indices.npy.
  2. Cast to float64.
  3. Standardize per dimension (zero mean, unit variance). Required: raw HSIC
     is scale-dependent, so without this the "dependence" numbers partly
     measure feature magnitude. Dimensions with zero variance are dropped.
  4. SVCCA  -> on PCA-k projections (k = 50 headline; 30 and 99%-variance as
                robustness checks). PCA is fit independently per representation
                on the analysis set. PCA(k) + CCA == SVCCA.
  5. CKA    -> on the full standardized feature space (standard usage in
                Kornblith et al.). Linear and RBF variants.
  6. HSIC   -> raw biased estimator, reported on BOTH the PCA-50 projections
                (comparable with the source paper) and the full space.

DIRECTIONALITY: every metric here is a SIMILARITY. Higher = more similar.
For the diversity reading, lower = more diverse. Do not put a "lower is
better" arrow on a CKA column without saying you mean lower similarity.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from . import cca_core
from . import cka_core

# Frozen protocol constants.
SVCCA_EPSILON = 1e-10          # stabilizer for the CCA solve; upstream default
                               # is 0.0, which leaves a singular solve
SVCCA_COMPONENTS = 50          # headline, matches the source paper
SVCCA_ROBUSTNESS_K = (30, 50)  # plus 99%-variance, handled separately
PCA_VARIANCE_TARGET = 0.99     # standard SVCCA truncation rule
RBF_THRESHOLD = 1.0            # median-distance heuristic multiplier


def standardize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-dimension zero mean, unit variance. Drops zero-variance dims.

    Args:
        x: [N, D] array.
        eps: variance floor below which a dimension is considered dead.

    Returns:
        [N, D'] float64 array with D' <= D.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected [N, D], got shape {x.shape}")
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    keep = sd.ravel() > eps
    if not keep.all():
        x = x[:, keep]
        mu = mu[:, keep]
        sd = sd[:, keep]
    return (x - mu) / sd


def pca_project(x: np.ndarray, n_components: int) -> np.ndarray:
    """Deterministic PCA projection, fit on x itself.

    svd_solver='full' avoids the randomized solver so results do not depend
    on any RNG state.
    """
    k = min(n_components, x.shape[1], x.shape[0])
    p = PCA(n_components=k, svd_solver="full")
    return p.fit_transform(x)


def pca_project_variance(x: np.ndarray,
                         target: float = PCA_VARIANCE_TARGET) -> np.ndarray:
    """PCA keeping enough components to explain `target` fraction of variance.

    This is the truncation rule used in the original SVCCA paper.
    """
    p = PCA(n_components=None, svd_solver="full")
    z = p.fit_transform(x)
    cum = np.cumsum(p.explained_variance_ratio_)
    k = int(np.searchsorted(cum, target) + 1)
    k = min(k, z.shape[1])
    return z[:, :k]


def svcca(x: np.ndarray,
          y: np.ndarray,
          n_components: int | None = SVCCA_COMPONENTS,
          variance_target: float | None = None) -> dict:
    """SVCCA between two standardized representations.

    Args:
        x, y: [N, D] standardized feature matrices. Row i must be the same
              example in both.
        n_components: fixed PCA truncation. Ignored if variance_target is set.
        variance_target: if given, truncate by explained variance instead.

    Returns:
        dict with:
          'mean'   : mean canonical correlation (the headline SVCCA number)
          'coefs'  : 1d array of canonical correlations, descending
          'k_x'    : components retained for x
          'k_y'    : components retained for y
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"row counts must match (same examples): {x.shape[0]} vs {y.shape[0]}")

    if variance_target is not None:
        zx = pca_project_variance(x, variance_target)
        zy = pca_project_variance(y, variance_target)
    else:
        zx = pca_project(x, n_components)
        zy = pca_project(y, n_components)

    # cca_core wants (num_neurons, num_datapoints) and asserts neurons <
    # datapoints, hence the transpose.
    ax, ay = zx.T.copy(), zy.T.copy()
    if ax.shape[0] >= ax.shape[1] or ay.shape[0] >= ay.shape[1]:
        raise ValueError(
            "need more examples than retained components; "
            f"got {ax.shape[0]}/{ay.shape[0]} components for {ax.shape[1]} examples")

    res = cca_core.get_cca_similarity(
        ax, ay,
        epsilon=SVCCA_EPSILON,
        compute_coefs=True,
        compute_dirns=False,
        verbose=False,
    )
    coefs = np.asarray(res["cca_coef1"], dtype=np.float64)

    # NOTE: we deliberately use a plain mean over ALL retained canonical
    # correlations, not res["mean"], which truncates at 98% cumulative mass
    # via sum_threshold. The plain mean is what the source paper reports and
    # what Figure 1 plots.
    return {
        "mean": float(np.mean(coefs)),
        "coefs": coefs,
        "k_x": int(zx.shape[1]),
        "k_y": int(zy.shape[1]),
    }


def spectrum_diagnostics(x: np.ndarray, k: int = SVCCA_COMPONENTS) -> dict:
    """How much of x survives PCA-k truncation, and its effective rank.

    These are REQUIRED diagnostics, not optional colour. Measured on synthetic
    data (see test_similarity.py CASE 4B), SVCCA-k is dominated by how well
    the top-k principal subspace captures the representation:

        top-50 variance   ~65%  ->  SVCCA of a linear reparam. only ~0.79
        top-50 variance   ~81%  ->  ~0.87
        top-50 variance  ~100%  ->  ~1.00

    So a low SVCCA number is only interpretable as "different representations"
    if the top-50 subspace actually holds most of the variance. If it does not,
    part of the measured dissimilarity is truncation artefact and must be
    reported as such.

    Effective rank is the participation ratio (sum L)^2 / sum(L^2) over the
    eigenvalues L of the correlation matrix. If effective rank falls well
    below k, the trailing components are noise directions that will not align
    across models and will drag SVCCA down for reasons unrelated to
    architecture.

    Returns:
        dict with 'topk_var_frac' and 'effective_rank'.
    """
    cov = np.cov(x.T)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    ev = np.clip(ev, 0.0, None)
    total = ev.sum()
    kk = min(k, len(ev))
    return {
        "topk_var_frac": float(ev[:kk].sum() / total) if total > 0 else 0.0,
        "effective_rank": float(total ** 2 / np.sum(ev ** 2)) if total > 0 else 0.0,
    }


def cka_and_hsic(x: np.ndarray, y: np.ndarray) -> dict:
    """Linear/RBF CKA (biased and debiased) plus raw biased HSIC.

    IMPORTANT -- biased CKA has a large positive finite-sample floor. Measured
    on independent gaussians at D=192 (test_similarity.py CASE 5):

        N=500   biased linear 0.275, RBF 0.391  |  debiased ~0.000
        N=1000  biased linear 0.162, RBF 0.246  |  debiased ~0.000
        N=2000  biased linear 0.087, RBF 0.139  |  debiased ~0.000
        N=5000  biased linear 0.037, RBF 0.061  |  debiased ~0.000

    At our N=2000 the biased RBF floor is 0.139, which is NOT negligible
    relative to plausible cross-architecture values. We therefore report the
    DEBIASED estimator as headline and keep the biased one only for
    comparability with papers that report it. Debiased CKA may be slightly
    negative; that is expected and means "no detectable dependence".

    Args:
        x, y: [N, D] standardized feature matrices, aligned by row.

    Returns:
        dict of scalars.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError("row counts must match (same examples)")

    gx_lin = cka_core.gram_linear(x)
    gy_lin = cka_core.gram_linear(y)
    gx_rbf = cka_core.gram_rbf(x, threshold=RBF_THRESHOLD)
    gy_rbf = cka_core.gram_rbf(y, threshold=RBF_THRESHOLD)

    return {
        "cka_linear": float(cka_core.cka(gx_lin, gy_lin)),
        "cka_rbf": float(cka_core.cka(gx_rbf, gy_rbf)),
        "cka_linear_debiased": float(cka_core.cka(gx_lin, gy_lin, debiased=True)),
        "cka_rbf_debiased": float(cka_core.cka(gx_rbf, gy_rbf, debiased=True)),
        "hsic_linear": cka_core.hsic_biased(gx_lin, gy_lin),
        "hsic_rbf": cka_core.hsic_biased(gx_rbf, gy_rbf),
    }


def compare(x_raw: np.ndarray, y_raw: np.ndarray) -> dict:
    """Full protocol-v1.0 comparison of two raw feature matrices.

    This is the single entry point run_analysis.py should call.

    Args:
        x_raw, y_raw: [N, D] raw (unstandardized) feature matrices, aligned
                      by row so that row i is the same example in both.

    Returns:
        Flat dict of metrics. Keys prefixed 'svcca50_' are the headline.
    """
    x = standardize(x_raw)
    y = standardize(y_raw)

    out: dict = {"n_examples": int(x.shape[0]),
                 "d_x": int(x.shape[1]),
                 "d_y": int(y.shape[1])}

    # --- SVCCA: headline at k=50, robustness at k=30 and 99% variance ---
    s50 = svcca(x, y, n_components=50)
    out["svcca50_mean"] = s50["mean"]
    out["svcca50_coefs"] = s50["coefs"]

    s30 = svcca(x, y, n_components=30)
    out["svcca30_mean"] = s30["mean"]

    s99 = svcca(x, y, variance_target=PCA_VARIANCE_TARGET)
    out["svcca99_mean"] = s99["mean"]
    out["svcca99_k_x"] = s99["k_x"]
    out["svcca99_k_y"] = s99["k_y"]

    # --- CKA on the full standardized space (standard usage) ---
    full = cka_and_hsic(x, y)
    for key in ("cka_linear", "cka_rbf", "cka_linear_debiased",
                "cka_rbf_debiased", "hsic_linear", "hsic_rbf"):
        out[f"{key}_full"] = full[key]

    # --- HSIC/CKA on PCA-50, for comparability with the source paper ---
    p50 = cka_and_hsic(pca_project(x, 50), pca_project(y, 50))
    for key in ("cka_linear", "cka_rbf", "cka_linear_debiased",
                "cka_rbf_debiased", "hsic_linear", "hsic_rbf"):
        out[f"{key}_pca50"] = p50[key]

    # --- spectrum diagnostics: required to interpret SVCCA at all ---
    dx = spectrum_diagnostics(x)
    dy = spectrum_diagnostics(y)
    out["top50_var_frac_x"] = dx["topk_var_frac"]
    out["top50_var_frac_y"] = dy["topk_var_frac"]
    out["effective_rank_x"] = dx["effective_rank"]
    out["effective_rank_y"] = dy["effective_rank"]

    return out