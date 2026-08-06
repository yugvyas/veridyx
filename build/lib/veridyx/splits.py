"""Train/validation/test partitioning, in two flavours that exist to be compared.

`grouped_split` keeps every near-duplicate cluster wholly inside one fold. It is the
split every reported Veridyx number uses.

`naive_split` is the ordinary row-level stratified split that most published work on
this dataset uses. It is kept, and evaluated, for one reason: the gap between the two
is the measurement. Reporting only the grouped number proves nothing to a reader who
has seen the inflated figures elsewhere; reporting both shows exactly how much of the
literature's performance is duplicate leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TRAIN, VAL, TEST = "train", "val", "test"
FOLDS = (TRAIN, VAL, TEST)
DEFAULT_RATIOS = {TRAIN: 0.70, VAL: 0.15, TEST: 0.15}


@dataclass(frozen=True)
class Split:
    """Fold assignment per row, plus the provenance needed to reproduce it."""

    fold: np.ndarray  # shape (n,), dtype '<U5', one of FOLDS
    kind: str  # "grouped" | "naive"
    seed: int

    def mask(self, fold: str) -> np.ndarray:
        if fold not in FOLDS:
            raise ValueError(f"unknown fold {fold!r}; expected one of {FOLDS}")
        return self.fold == fold

    def summary(self, labels: np.ndarray) -> dict[str, dict[str, float]]:
        """Row count and fraud rate per fold — the sanity check worth printing."""
        out: dict[str, dict[str, float]] = {}
        for f in FOLDS:
            m = self.mask(f)
            n = int(m.sum())
            out[f] = {
                "n": n,
                "n_fraud": int(labels[m].sum()),
                "fraud_rate": float(labels[m].mean()) if n else 0.0,
            }
        return out


def _targets(n: int, ratios: dict[str, float]) -> dict[str, float]:
    return {f: n * ratios[f] for f in FOLDS}


def grouped_split(
    clusters: np.ndarray,
    labels: np.ndarray,
    seed: int = 0,
    ratios: dict[str, float] | None = None,
) -> Split:
    """Assign whole near-duplicate clusters to folds, balancing size and fraud rate.

    Greedy, largest-cluster-first. Each cluster goes to whichever fold is currently
    furthest behind its target, scored on total rows and fraud rows together. Exact
    stratification is impossible once clusters are indivisible — a 40-posting scam
    campaign lands somewhere whole — so the goal is the best achievable balance, and
    `Split.summary` reports what was actually achieved rather than what was intended.

    Deterministic given `seed`: ties among equal-size clusters are broken by a seeded
    permutation, never by dict or set ordering.
    """
    ratios = ratios or DEFAULT_RATIOS
    n = len(clusters)
    if n != len(labels):
        raise ValueError("clusters and labels must be the same length")

    cluster_ids, inverse = np.unique(clusters, return_inverse=True)
    rng = np.random.default_rng(seed)
    order_jitter = rng.permutation(len(cluster_ids))

    sizes = np.bincount(inverse, minlength=len(cluster_ids))
    frauds = np.bincount(inverse, weights=labels.astype(float), minlength=len(cluster_ids))

    # Largest first; jitter breaks size ties deterministically.
    order = sorted(
        range(len(cluster_ids)),
        key=lambda c: (-int(sizes[c]), -float(frauds[c]), int(order_jitter[c])),
    )

    size_target = _targets(n, ratios)
    fraud_target = _targets(float(labels.sum()), ratios)
    size_have = dict.fromkeys(FOLDS, 0.0)
    fraud_have = dict.fromkeys(FOLDS, 0.0)
    assignment: dict[int, str] = {}

    for c in order:
        # Deficit measured in units of each fold's own target, so a fold that should
        # hold 70% is not permanently judged "behind" a fold that should hold 15%.
        def deficit(f: str, c: int = c) -> float:
            size_gap = (size_target[f] - size_have[f]) / max(size_target[f], 1.0)
            fraud_gap = (fraud_target[f] - fraud_have[f]) / max(fraud_target[f], 1.0)
            # Fraud is the scarce quantity (~5% of rows); weight it to match.
            return size_gap + 2.0 * fraud_gap

        best = max(FOLDS, key=deficit)
        assignment[c] = best
        size_have[best] += float(sizes[c])
        fraud_have[best] += float(frauds[c])

    fold = np.array([assignment[c] for c in inverse], dtype="<U5")
    return Split(fold=fold, kind="grouped", seed=seed)


def naive_split(
    labels: np.ndarray,
    seed: int = 0,
    ratios: dict[str, float] | None = None,
) -> Split:
    """Ordinary row-level stratified split, ignoring duplicates entirely.

    This is the leaky baseline. It exists only so the leak can be quantified; nothing
    in Veridyx should be *reported* from it without the grouped number beside it.
    """
    ratios = ratios or DEFAULT_RATIOS
    rng = np.random.default_rng(seed)
    fold = np.empty(len(labels), dtype="<U5")

    for value in (0, 1):
        idx = np.flatnonzero(labels.astype(int) == value)
        rng.shuffle(idx)
        n_train = round(len(idx) * ratios[TRAIN])
        n_val = round(len(idx) * ratios[VAL])
        fold[idx[:n_train]] = TRAIN
        fold[idx[n_train : n_train + n_val]] = VAL
        fold[idx[n_train + n_val :]] = TEST

    return Split(fold=fold, kind="naive", seed=seed)
