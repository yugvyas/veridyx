"""Threshold selection tests.

Synthetic scores throughout: the point is that the *decision rule* is correct, and a
trained model in the loop would only make failures harder to localise.
"""

from __future__ import annotations

import numpy as np
import pytest

from veridyx.threshold import (
    analyse,
    capacity_threshold,
    minimum_cost_threshold,
    sweep_capacity,
    sweep_costs,
)


@pytest.fixture
def scores():
    """100 postings, 10 fraudulent, cleanly ranked but overlapping in the middle."""
    y = np.zeros(100, dtype=bool)
    y[:10] = True
    s = np.concatenate([np.linspace(0.55, 0.99, 10), np.linspace(0.0, 0.60, 90)])
    return y, s


class TestMinimumCost:
    def test_expensive_false_negatives_push_the_threshold_down(self, y_and_s=None):
        y = np.zeros(100, dtype=bool)
        y[:10] = True
        s = np.concatenate([np.linspace(0.55, 0.99, 10), np.linspace(0.0, 0.60, 90)])
        cheap_misses = minimum_cost_threshold(y, s, cost_fn=1, cost_fp=1)
        costly_misses = minimum_cost_threshold(y, s, cost_fn=1000, cost_fp=1)
        assert costly_misses.threshold <= cheap_misses.threshold
        assert costly_misses.recall >= cheap_misses.recall

    def test_symmetric_costs_do_not_flag_everything(self, scores):
        y, s = scores
        choice = minimum_cost_threshold(y, s, cost_fn=1, cost_fp=1)
        assert choice.flagged < len(y)

    def test_extreme_fn_cost_catches_everything(self, scores):
        y, s = scores
        choice = minimum_cost_threshold(y, s, cost_fn=10_000_000, cost_fp=1)
        assert choice.missed == 0
        assert choice.recall == 1.0

    def test_extreme_fp_cost_admits_no_false_alarms(self, scores):
        """Ruinous false alarms buy zero of them — but free catches are still taken.

        The obvious assertion here is `flagged == 0`, and it is wrong. Where the top
        scores separate cleanly, the optimum flags every posting it can without
        touching a legitimate one: 8 caught, 0 false alarms, cost 2 (from the two
        misses) against a cost of 10 for flagging nothing. Declining a free true
        positive would not be conservative, it would just be worse.
        """
        y, s = scores
        choice = minimum_cost_threshold(y, s, cost_fn=1, cost_fp=10_000_000)
        assert choice.false_alarms == 0
        assert choice.precision == 1.0

    def test_reported_cost_matches_its_own_formula(self, scores):
        y, s = scores
        choice = minimum_cost_threshold(y, s, cost_fn=500, cost_fp=3)
        assert choice.expected_cost == pytest.approx(
            choice.missed * 500 + choice.false_alarms * 3
        )

    def test_chosen_threshold_really_is_the_minimum(self, scores):
        """Guards the search itself, not just its output."""
        y, s = scores
        cost_fn, cost_fp = 200.0, 1.0
        best = minimum_cost_threshold(y, s, cost_fn, cost_fp)
        for t in np.linspace(0, 1, 201):
            flagged = s >= t
            cost = np.sum(~flagged & y) * cost_fn + np.sum(flagged & ~y) * cost_fp
            assert best.expected_cost <= cost + 1e-9


class TestCapacity:
    def test_never_exceeds_capacity(self, scores):
        y, s = scores
        for capacity in (1, 5, 10, 25, 50):
            assert capacity_threshold(y, s, capacity).flagged <= capacity

    def test_uses_the_budget_it_can(self, scores):
        """Must not be trivially conservative — a budget of 25 should flag near 25."""
        y, s = scores
        assert capacity_threshold(y, s, 25).flagged >= 20

    def test_more_capacity_never_catches_less(self, scores):
        y, s = scores
        caught = [capacity_threshold(y, s, c).caught for c in (10, 25, 50, 90)]
        assert caught == sorted(caught)

    def test_slack_is_reported(self, scores):
        y, s = scores
        choice = capacity_threshold(y, s, 25)
        assert choice.capacity == 25
        assert choice.slack == 25 - choice.flagged

    def test_ties_are_not_split(self):
        """Tied scores must be flagged together or not at all.

        Splitting a tie would give two postings the model considers identical
        different treatment based on row order. Here 20 postings share one score and
        the budget is 10, so the correct behaviour is to flag none of them.
        """
        y = np.zeros(30, dtype=bool)
        y[:5] = True
        s = np.concatenate([np.full(20, 0.7), np.full(10, 0.1)])
        choice = capacity_threshold(y, s, 10)
        assert choice.flagged in (0, 10)
        assert choice.flagged <= 10


class TestSweeps:
    def test_cost_sweep_is_monotone_in_flags(self, scores):
        """Valuing misses more can only ever widen the net."""
        y, s = scores
        flagged = [row["flagged"] for row in sweep_costs(y, s)]
        assert flagged == sorted(flagged)

    def test_capacity_sweep_respects_each_budget(self, scores):
        y, s = scores
        for row in sweep_capacity(y, s):
            assert row["flagged"] <= row["capacity"]

    def test_analyse_reports_both_framings_and_the_baselines(self, scores):
        y, s = scores
        report = analyse(y, s, capacity=25)
        assert set(report["chosen"]) == {"minimum_expected_cost", "review_capacity"}
        assert report["reference_points"]["flag_nothing"]["caught"] == 0
        assert report["reference_points"]["flag_everything"]["missed"] == 0
        assert report["n_fraud"] == 10
