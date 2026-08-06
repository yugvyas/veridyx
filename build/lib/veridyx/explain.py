"""Per-posting attribution, so every flag can be defended.

A flag a reviewer cannot interrogate is a flag they will either rubber-stamp or
ignore, and both defeat the point of having a human in the loop. Every `Verdict`
Veridyx emits carries the handful of features that actually moved the score.

**On not using the `shap` package for the GBM.** `shap.TreeExplainer` materialises a
dense `(n_rows, n_features)` array. At this vocabulary that is 2,554 x 30,020 floats —
613 MB for one test fold, and it grows with the vocabulary that Day 2 showed we want
large. LightGBM computes the identical exact TreeSHAP values natively via
`predict(..., pred_contrib=True)`, accepts the sparse design matrix directly, and is
batched here so memory stays bounded regardless of fold size.

For the linear baseline no approximation is needed at all: the exact SHAP value of
feature *i* is `coef_i * (x_i - E[x_i])`, computed directly.

The two paths return the same `Contribution` type, so downstream code — the review
sheet, the endpoint, the drift report — never branches on model architecture.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import sparse

from veridyx.features import FeatureSet
from veridyx.models.base import Model
from veridyx.models.baseline import TfidfLogisticRegression
from veridyx.models.gbm import TFIDF_PREFIX, GradientBoosting
from veridyx.schema import Contribution

log = logging.getLogger(__name__)

# Rows per batch when computing tree contributions. 200 x 30,020 floats is ~48 MB,
# which keeps peak memory flat while staying large enough that the per-call overhead
# does not dominate.
CONTRIB_BATCH = 200

# Rows sampled when estimating global importance. Mean |SHAP| converges quickly; the
# ranking is stable well before the full fold is used.
GLOBAL_SAMPLE = 500


class ExplanationError(RuntimeError):
    """Raised when a model has no attribution path."""


def _tidy(name: str) -> str:
    """Strip the internal TF-IDF prefix for display."""
    return name[len(TFIDF_PREFIX) :] if name.startswith(TFIDF_PREFIX) else name


def _top_k(names: list[str], values: np.ndarray, k: int) -> list[Contribution]:
    """The k features with the largest absolute push, ordered by that magnitude.

    Absolute value, not signed: a strong push *toward legitimate* is exactly what a
    reviewer needs to see when they are deciding whether to dismiss a borderline flag.
    """
    if values.size == 0:
        return []
    order = np.argsort(np.abs(values))[::-1][:k]
    return [
        Contribution(feature=_tidy(names[i]), value=float(values[i]))
        for i in order
        if values[i] != 0.0
    ]


# --------------------------------------------------------------------------------
# Per-model contribution matrices
# --------------------------------------------------------------------------------


def _gbm_contributions(model: GradientBoosting, features: FeatureSet) -> np.ndarray:
    """Exact TreeSHAP from LightGBM, batched. Shape (n, n_features).

    LightGBM mirrors the sparsity of its input: given the sparse design matrix it
    returns a sparse contribution matrix, and `np.asarray` on that yields a
    zero-dimensional object array rather than the 2-D array the shape suggests. Each
    batch is densified explicitly, which is why `CONTRIB_BATCH` exists — the dense
    form of a whole fold would be 613 MB.
    """
    design = model.design_matrix(features)
    n = design.shape[0]
    out = np.zeros((n, len(model.feature_names)), dtype=np.float64)
    for start in range(0, n, CONTRIB_BATCH):
        chunk = design[start : start + CONTRIB_BATCH]
        contrib = model.classifier.booster_.predict(chunk, pred_contrib=True)
        dense = contrib.toarray() if sparse.issparse(contrib) else np.asarray(contrib)
        # The final column is the expected value (base rate), not a feature.
        out[start : start + chunk.shape[0]] = dense[:, :-1]
    return out


def _linear_contributions(
    model: TfidfLogisticRegression, features: FeatureSet
) -> tuple[np.ndarray, list[str]]:
    """Exact SHAP for a linear model: `coef_i * (x_i - mean_i)`.

    Computed as `x_i * coef_i` minus the constant row `coef_i * mean_i` rather than
    by centring first. Sparse matrices do not broadcast a (1, vocab) row against an
    (n, vocab) matrix — `matrix - csr_matrix(means)` raises "inconsistent shapes" —
    and centring a sparse matrix would densify it in place regardless, since almost
    every mean is non-zero.

    Batched for the same reason as the tree path: at a 50,000-term vocabulary a
    500-row global-importance sample is 200 MB dense.
    """
    matrix = model.vectorizer.transform(features.texts)
    coefs = model.classifier.coef_[0]
    means = np.asarray(matrix.mean(axis=0)).ravel()
    baseline = coefs * means

    n = matrix.shape[0]
    out = np.zeros((n, matrix.shape[1]), dtype=np.float64)
    for start in range(0, n, CONTRIB_BATCH):
        chunk = matrix[start : start + CONTRIB_BATCH]
        out[start : start + chunk.shape[0]] = chunk.multiply(coefs).toarray() - baseline
    return out, list(model.vectorizer.get_feature_names_out())


def contribution_matrix(model: Model, features: FeatureSet) -> tuple[np.ndarray, list[str]]:
    """Signed per-feature contributions and the matching feature names."""
    if isinstance(model, GradientBoosting):
        return _gbm_contributions(model, features), list(model.feature_names)
    if isinstance(model, TfidfLogisticRegression):
        return _linear_contributions(model, features)
    raise ExplanationError(
        f"{type(model).__name__} has no attribution path. The GBM and the linear "
        "baseline are explainable by construction; DistilBERT is the ceiling check "
        "and is deliberately not the deployed model."
    )


# --------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------


def explain(model: Model, features: FeatureSet, k: int = 8) -> list[list[Contribution]]:
    """Top-k contributions for every row in `features`.

    Intended for the flagged subset, not a whole fold: explaining 40 flags is the
    real workload, and explaining 2,554 postings nobody will read is not.
    """
    matrix, names = contribution_matrix(model, features)
    return [_top_k(names, matrix[i], k) for i in range(matrix.shape[0])]


def explain_one(model: Model, features: FeatureSet, index: int = 0, k: int = 8) -> list[Contribution]:
    """Contributions for a single posting."""
    single = FeatureSet(
        matrix=features.matrix[index : index + 1],
        names=features.names,
        texts=features.texts[index : index + 1],
        regime=features.regime,
    )
    return explain(model, single, k)[0]


def global_importance(
    model: Model, features: FeatureSet, k: int = 25, sample: int = GLOBAL_SAMPLE, seed: int = 0
) -> list[Contribution]:
    """Mean |SHAP| per feature over a sample — the global explanation.

    Values are mean absolute contributions, so they are magnitudes rather than signed
    pushes; `Contribution.direction()` is not meaningful on this output and callers
    should not use it here.
    """
    n = len(features.texts)
    idx = (
        np.random.default_rng(seed).choice(n, size=sample, replace=False)
        if n > sample
        else np.arange(n)
    )
    subset = FeatureSet(
        matrix=features.matrix[idx],
        names=features.names,
        texts=[features.texts[i] for i in idx],
        regime=features.regime,
    )
    matrix, names = contribution_matrix(model, subset)
    mean_abs = np.abs(matrix).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:k]
    return [
        Contribution(feature=_tidy(names[i]), value=float(mean_abs[i]))
        for i in order
        if mean_abs[i] > 0
    ]
