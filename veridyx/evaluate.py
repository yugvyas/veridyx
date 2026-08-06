"""Metrics, and the experiment matrix that produces every reported number.

Accuracy is not computed anywhere in this file, on purpose. At a 4.84% fraud rate a
model that answers "legitimate" to everything scores 95.2% accurate and catches
nothing. The metrics here — precision, recall, F1, and above all PR-AUC — are the ones
that can tell those two models apart.

Two methodological points that are easy to get wrong and that decide whether the
results table means anything:

1. **The decision threshold is chosen on validation and applied to test.** Choosing it
   on test and then reporting test F1 is the most common way to report a number that
   cannot be reproduced in deployment. PR-AUC is threshold-free and is reported
   alongside precisely so the comparison does not rest on threshold choice at all.

2. **Every cell of the matrix uses the same code path.** Model, regime and split kind
   vary; nothing else does.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from veridyx import features as feat
from veridyx.data import ROOT, prepare
from veridyx.features import FeatureSet
from veridyx.models.base import Model
from veridyx.models.baseline import TfidfLogisticRegression
from veridyx.models.gbm import GradientBoosting
from veridyx.splits import TEST, TRAIN, VAL

log = logging.getLogger(__name__)

EXPERIMENTS_DIR = ROOT / "experiments"
RESULTS_FILE = EXPERIMENTS_DIR / "results.json"

MODEL_FACTORIES: dict[str, type[Model]] = {
    TfidfLogisticRegression.name: TfidfLogisticRegression,
    GradientBoosting.name: GradientBoosting,
}


# --------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """The threshold maximising F1, and that F1. Intended for VALIDATION scores only.

    `precision_recall_curve` returns one fewer threshold than precision/recall points
    (the final point is recall=0, precision=1, with no corresponding threshold), so the
    arrays are trimmed before the argmax. Getting this off-by-one wrong silently
    returns a threshold one position out of alignment with the F1 it claims to
    maximise.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    precision, recall = precision[:-1], recall[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(
            (precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0
        )
    if f1.size == 0:
        return 0.5, 0.0
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def metrics_at(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    """The full metric set at a fixed threshold, plus the threshold-free ones."""
    y_pred = y_score >= threshold
    tp = int(np.sum(y_pred & y_true))
    fp = int(np.sum(y_pred & ~y_true))
    fn = int(np.sum(~y_pred & y_true))
    tn = int(np.sum(~y_pred & ~y_true))

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(threshold),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "n": len(y_true),
        "n_positive": int(y_true.sum()),
        # Reported so a reader can see for themselves what accuracy hides here: the
        # all-legitimate model's accuracy is 1 - positive_rate, typically ~0.95.
        "positive_rate": float(y_true.mean()),
        "accuracy_of_predicting_all_legitimate": float(1 - y_true.mean()),
    }


# --------------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------------


@dataclass
class RunResult:
    model: str
    regime: str
    split_kind: str
    seed: int
    test: dict
    validation: dict
    train_seconds: float
    scores: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    labels: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))

    @property
    def key(self) -> str:
        return f"{self.model}/{self.regime}/{self.split_kind}"

    def to_json(self) -> dict:
        payload = asdict(self)
        payload.pop("scores")
        payload.pop("labels")
        return payload


def _subset(features: FeatureSet, mask: np.ndarray) -> FeatureSet:
    idx = np.flatnonzero(mask)
    return FeatureSet(
        matrix=features.matrix[idx],
        names=features.names,
        texts=[features.texts[i] for i in idx],
        regime=features.regime,
    )


def run_one(
    model_name: str,
    features: FeatureSet,
    labels: np.ndarray,
    fold: np.ndarray,
    split_kind: str,
    seed: int = 0,
) -> RunResult:
    """Train on train, pick a threshold on validation, report on test."""
    model = MODEL_FACTORIES[model_name](seed=seed)

    train_mask, val_mask, test_mask = (fold == TRAIN, fold == VAL, fold == TEST)
    started = time.perf_counter()
    model.fit(_subset(features, train_mask), labels[train_mask])
    train_seconds = time.perf_counter() - started

    val_scores = model.predict_proba(_subset(features, val_mask))
    threshold, _ = best_f1_threshold(labels[val_mask], val_scores)

    test_scores = model.predict_proba(_subset(features, test_mask))

    return RunResult(
        model=model_name,
        regime=features.regime,
        split_kind=split_kind,
        seed=seed,
        test=metrics_at(labels[test_mask], test_scores, threshold),
        validation=metrics_at(labels[val_mask], val_scores, threshold),
        train_seconds=round(train_seconds, 2),
        scores=test_scores,
        labels=labels[test_mask],
    )


# --------------------------------------------------------------------------------
# The matrix
# --------------------------------------------------------------------------------


def run_matrix(
    seed: int = 0,
    models: list[str] | None = None,
    regimes: list[str] | None = None,
    split_kinds: list[str] | None = None,
) -> list[RunResult]:
    """Every (model x regime x split) cell, from one shared dataset preparation."""
    models = models or list(MODEL_FACTORIES)
    regimes = regimes or list(feat.REGIMES)
    split_kinds = split_kinds or ["grouped", "naive"]

    dataset = prepare(seed=seed)
    labels = dataset.labels

    results: list[RunResult] = []
    for regime in regimes:
        # Feature construction is the expensive part; do it once per regime and reuse
        # it across every model and split kind so the cells cannot diverge.
        features = feat.build(regime, dataset.postings)
        for split_kind in split_kinds:
            fold = dataset.frame[f"fold_{split_kind}"].to_numpy()
            for model_name in models:
                log.info("running %s / %s / %s", model_name, regime, split_kind)
                result = run_one(model_name, features, labels, fold, split_kind, seed)
                log.info(
                    "  PR-AUC %.4f  F1 %.4f  P %.4f  R %.4f  (%.1fs)",
                    result.test["pr_auc"],
                    result.test["f1"],
                    result.test["precision"],
                    result.test["recall"],
                    result.train_seconds,
                )
                results.append(result)
    return results


def leakage_delta(results: list[RunResult]) -> list[dict]:
    """Naive minus grouped, per (model, regime). The headline comparison.

    A large positive delta means the naive split reported a model as better than it is,
    and the size of that gap is how much of the published performance on this dataset
    is duplicate memorisation rather than generalisation.
    """
    by_key = {(r.model, r.regime, r.split_kind): r for r in results}
    rows = []
    for model, regime, split_kind in list(by_key):
        if split_kind != "grouped":
            continue
        grouped = by_key[(model, regime, "grouped")]
        naive = by_key.get((model, regime, "naive"))
        if naive is None:
            continue
        rows.append(
            {
                "model": model,
                "regime": regime,
                "grouped_pr_auc": grouped.test["pr_auc"],
                "naive_pr_auc": naive.test["pr_auc"],
                "pr_auc_inflation": round(naive.test["pr_auc"] - grouped.test["pr_auc"], 4),
                "grouped_f1": grouped.test["f1"],
                "naive_f1": naive.test["f1"],
                "f1_inflation": round(naive.test["f1"] - grouped.test["f1"], 4),
            }
        )
    return sorted(rows, key=lambda r: (r["regime"], r["model"]))


def regime_gap(results: list[RunResult]) -> list[dict]:
    """FULL minus PORTABLE on the grouped split.

    How much of the benchmark score comes from columns no live feed provides. This is
    the number that decides whether a deployed model can be expected to behave like
    the one in the results table.
    """
    by_key = {(r.model, r.regime, r.split_kind): r for r in results}
    rows = []
    for model in {m for m, _, _ in by_key}:
        full = by_key.get((model, feat.FULL, "grouped"))
        portable = by_key.get((model, feat.PORTABLE, "grouped"))
        if not (full and portable):
            continue
        rows.append(
            {
                "model": model,
                "full_pr_auc": full.test["pr_auc"],
                "portable_pr_auc": portable.test["pr_auc"],
                "benchmark_only_advantage": round(
                    full.test["pr_auc"] - portable.test["pr_auc"], 4
                ),
                "full_f1": full.test["f1"],
                "portable_f1": portable.test["f1"],
            }
        )
    return sorted(rows, key=lambda r: r["model"])


def write_results(results: list[RunResult], path: Path | None = None) -> Path:
    path = path or RESULTS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "runs": [r.to_json() for r in results],
        "leakage_delta": leakage_delta(results),
        "regime_gap": regime_gap(results),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    log.info("wrote %s", path)
    return path


def _print_table(results: list[RunResult]) -> None:
    print(f"\n{'model':14s} {'regime':9s} {'split':8s} {'PR-AUC':>8s} {'F1':>7s} {'P':>7s} {'R':>7s}")
    print("-" * 64)
    for r in sorted(results, key=lambda r: (r.regime, r.split_kind, r.model)):
        t = r.test
        print(
            f"{r.model:14s} {r.regime:9s} {r.split_kind:8s} "
            f"{t['pr_auc']:8.4f} {t['f1']:7.4f} {t['precision']:7.4f} {t['recall']:7.4f}"
        )

    print("\nleakage: how much the naive split inflates each number")
    print(f"{'model':14s} {'regime':9s} {'grouped':>9s} {'naive':>9s} {'inflation':>10s}")
    print("-" * 56)
    for row in leakage_delta(results):
        print(
            f"{row['model']:14s} {row['regime']:9s} {row['grouped_pr_auc']:9.4f} "
            f"{row['naive_pr_auc']:9.4f} {row['pr_auc_inflation']:+10.4f}"
        )

    print("\nregime gap: PR-AUC bought by columns no live feed has (grouped split)")
    print(f"{'model':14s} {'FULL':>9s} {'PORTABLE':>9s} {'advantage':>10s}")
    print("-" * 46)
    for row in regime_gap(results):
        print(
            f"{row['model']:14s} {row['full_pr_auc']:9.4f} "
            f"{row['portable_pr_auc']:9.4f} {row['benchmark_only_advantage']:+10.4f}"
        )


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Veridyx evaluation matrix.")
    parser.add_argument("--models", nargs="*", choices=list(MODEL_FACTORIES))
    parser.add_argument("--regimes", nargs="*", choices=list(feat.REGIMES))
    parser.add_argument("--splits", nargs="*", choices=["grouped", "naive"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    results = run_matrix(
        seed=args.seed,
        models=args.models,
        regimes=args.regimes,
        split_kinds=args.splits,
    )
    _print_table(results)
    write_results(results, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
