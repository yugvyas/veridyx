"""Metric and harness tests.

These use synthetic scores rather than a trained model so that a metric bug cannot
hide behind a model change, and so the suite stays fast enough to run on every commit.
"""

from __future__ import annotations

import numpy as np
import pytest

from veridyx.evaluate import (
    RunResult,
    best_f1_threshold,
    leakage_delta,
    metrics_at,
    regime_gap,
)


class TestMetrics:
    def test_perfect_separation(self):
        y = np.array([0, 0, 1, 1], dtype=bool)
        s = np.array([0.1, 0.2, 0.8, 0.9])
        m = metrics_at(y, s, 0.5)
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0
        assert m["pr_auc"] == pytest.approx(1.0)

    def test_all_legitimate_predictor_scores_zero_not_high(self):
        """The failure this project exists to avoid, asserted directly.

        A model that never predicts fraud is 95% accurate on this class balance and
        must score 0 on every metric Veridyx reports.
        """
        y = np.zeros(100, dtype=bool)
        y[:5] = True
        s = np.zeros(100)  # never flags anything
        m = metrics_at(y, s, 0.5)
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0
        assert m["accuracy_of_predicting_all_legitimate"] == pytest.approx(0.95)

    def test_confusion_counts_sum_to_n(self):
        rng = np.random.default_rng(0)
        y = rng.random(200) < 0.05
        s = rng.random(200)
        m = metrics_at(y, s, 0.5)
        total = (
            m["true_positives"]
            + m["false_positives"]
            + m["false_negatives"]
            + m["true_negatives"]
        )
        assert total == m["n"] == 200

    def test_threshold_is_inclusive(self):
        y = np.array([0, 1], dtype=bool)
        s = np.array([0.5, 0.5])
        m = metrics_at(y, s, 0.5)
        assert m["true_positives"] == 1
        assert m["false_positives"] == 1


class TestThresholdSelection:
    def test_recovers_a_clean_boundary(self):
        y = np.array([0] * 50 + [1] * 50, dtype=bool)
        s = np.concatenate([np.linspace(0.0, 0.4, 50), np.linspace(0.6, 1.0, 50)])
        threshold, f1 = best_f1_threshold(y, s)
        assert f1 == pytest.approx(1.0)
        assert 0.4 < threshold <= 0.6

    def test_threshold_and_f1_agree(self):
        """Guards the precision_recall_curve off-by-one.

        The curve returns one more precision/recall point than thresholds. Trimming
        the wrong end returns a threshold misaligned with the F1 it reports — the
        numbers still look reasonable, which is what makes it dangerous.
        """
        rng = np.random.default_rng(1)
        y = rng.random(500) < 0.1
        s = np.clip(y * 0.5 + rng.normal(0.3, 0.2, 500), 0, 1)
        threshold, claimed_f1 = best_f1_threshold(y, s)
        assert metrics_at(y, s, threshold)["f1"] == pytest.approx(claimed_f1, abs=1e-9)

    def test_degenerate_input_returns_a_usable_default(self):
        y = np.zeros(10, dtype=bool)
        threshold, f1 = best_f1_threshold(y, np.zeros(10))
        assert 0.0 <= threshold <= 1.0
        assert f1 == 0.0


def _result(model, regime, split_kind, pr_auc, f1) -> RunResult:
    metrics = {"pr_auc": pr_auc, "f1": f1}
    return RunResult(
        model=model,
        regime=regime,
        split_kind=split_kind,
        seed=0,
        test=metrics,
        validation=metrics,
        train_seconds=0.0,
    )


class TestComparisons:
    def test_leakage_delta_is_naive_minus_grouped(self):
        results = [
            _result("m", "portable", "grouped", 0.70, 0.60),
            _result("m", "portable", "naive", 0.80, 0.68),
        ]
        (row,) = leakage_delta(results)
        assert row["pr_auc_inflation"] == pytest.approx(0.10)
        assert row["f1_inflation"] == pytest.approx(0.08)

    def test_leakage_delta_skips_incomplete_pairs(self):
        assert leakage_delta([_result("m", "portable", "grouped", 0.7, 0.6)]) == []

    def test_regime_gap_is_full_minus_portable(self):
        results = [
            _result("m", "full", "grouped", 0.88, 0.79),
            _result("m", "portable", "grouped", 0.77, 0.70),
        ]
        (row,) = regime_gap(results)
        assert row["benchmark_only_advantage"] == pytest.approx(0.11)

    def test_regime_gap_uses_grouped_only(self):
        """A gap computed on the leaky split would confound the two effects."""
        results = [
            _result("m", "full", "naive", 0.95, 0.90),
            _result("m", "portable", "naive", 0.85, 0.80),
        ]
        assert regime_gap(results) == []
