"""Split integrity.

The first test in this file is the one the entire rigor argument rests on. If a
near-duplicate cluster can span folds, every grouped metric Veridyx reports is as
leaky as the literature it criticises.
"""

from __future__ import annotations

import numpy as np
import pytest

from veridyx.splits import FOLDS, grouped_split, naive_split


def _campaign_data(n_singletons=900, n_campaigns=12, campaign_size=25, seed=7):
    """Synthetic EMSCAD shape: mostly unique postings, plus a few scam campaigns.

    Campaigns are all-fraud, which is the realistic and adversarial case: they are
    exactly the rows a leaky split duplicates across the boundary.
    """
    rng = np.random.default_rng(seed)
    clusters, labels = [], []
    for i in range(n_singletons):
        clusters.append(i)
        labels.append(rng.random() < 0.02)
    next_id = n_singletons
    for _ in range(n_campaigns):
        for _ in range(campaign_size):
            clusters.append(next_id)
            labels.append(True)
        next_id += 1
    return np.array(clusters), np.array(labels, dtype=bool)


class TestGroupedSplit:
    def test_no_cluster_spans_folds(self):
        """THE invariant. Every member of a near-duplicate cluster shares a fold."""
        clusters, labels = _campaign_data()
        split = grouped_split(clusters, labels, seed=0)

        for cid in np.unique(clusters):
            folds_seen = set(split.fold[clusters == cid])
            assert len(folds_seen) == 1, (
                f"cluster {cid} was split across {sorted(folds_seen)} — "
                "near-duplicates are leaking between train and test"
            )

    def test_every_row_assigned(self):
        clusters, labels = _campaign_data()
        split = grouped_split(clusters, labels, seed=0)
        assert set(split.fold) <= set(FOLDS)
        assert len(split.fold) == len(clusters)
        assert (split.fold != "").all()

    def test_deterministic_for_a_seed(self):
        clusters, labels = _campaign_data()
        a = grouped_split(clusters, labels, seed=3)
        b = grouped_split(clusters, labels, seed=3)
        assert (a.fold == b.fold).all()

    def test_seed_actually_changes_assignment(self):
        """Guards against a seed that is accepted and then ignored."""
        clusters, labels = _campaign_data(n_singletons=400)
        a = grouped_split(clusters, labels, seed=1)
        b = grouped_split(clusters, labels, seed=99)
        assert not (a.fold == b.fold).all()

    def test_every_fold_gets_some_fraud(self):
        """A fold with zero positives makes PR-AUC undefined and the run worthless."""
        clusters, labels = _campaign_data()
        split = grouped_split(clusters, labels, seed=0)
        for fold, stats in split.summary(labels).items():
            assert stats["n_fraud"] > 0, f"fold {fold} has no fraudulent rows"

    def test_fold_sizes_are_roughly_on_target(self):
        """Indivisible clusters make exact ratios impossible; require close, not exact."""
        clusters, labels = _campaign_data()
        split = grouped_split(clusters, labels, seed=0)
        summary = split.summary(labels)
        n = len(labels)
        assert 0.60 * n <= summary["train"]["n"] <= 0.78 * n
        for fold in ("val", "test"):
            assert 0.09 * n <= summary[fold]["n"] <= 0.22 * n

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError):
            grouped_split(np.arange(10), np.zeros(9, dtype=bool))


class TestNaiveSplit:
    def test_all_rows_assigned_exactly_once(self):
        _, labels = _campaign_data()
        split = naive_split(labels, seed=0)
        assert len(split.fold) == len(labels)
        assert set(split.fold) == set(FOLDS)

    def test_stratified_on_label(self):
        _, labels = _campaign_data()
        summary = naive_split(labels, seed=0).summary(labels)
        overall = labels.mean()
        for fold in FOLDS:
            assert summary[fold]["fraud_rate"] == pytest.approx(overall, abs=0.02)

    def test_it_does_leak_clusters(self):
        """Documents the contrast the comparison depends on.

        This is not a defect being tolerated — it is the leaky baseline whose failure
        mode Veridyx exists to measure. If this test ever stops failing to contain
        clusters, `naive_split` has quietly become `grouped_split` and the reported
        delta would collapse to zero for the wrong reason.
        """
        clusters, labels = _campaign_data()
        split = naive_split(labels, seed=0)
        spanning = sum(
            1 for cid in np.unique(clusters) if len(set(split.fold[clusters == cid])) > 1
        )
        assert spanning > 0
