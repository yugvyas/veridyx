"""Choosing the decision threshold as a cost decision, not a metric convenience.

Maximising F1 treats a missed fraud and a false alarm as equally bad. They are not.
A missed fraudulent posting can cost a candidate money, documents, or an identity; a
false alarm costs a reviewer a minute. Any threshold that ignores that asymmetry is
answering a question nobody asked.

Two framings, because deployments come in two shapes:

**Expected cost.** Given a cost per false negative and a cost per false positive,
pick the threshold minimising total expected cost. Honest, and completely dependent
on numbers that are hard to defend — so `sweep_costs` reports how the answer moves
across a wide range of ratios rather than resting on one guess.

**Review capacity.** A reviewer can look at K postings a day. Given that hard ceiling,
pick the threshold that catches the most fraud within it. This needs no cost estimate
at all, only a headcount, which is why it is usually the framing an operator can
actually act on.

The two agree less often than you would expect, and where they disagree is the
interesting part of the slide.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, replace

import numpy as np

from veridyx.data import ROOT

log = logging.getLogger(__name__)

EXPERIMENTS_DIR = ROOT / "experiments"
THRESHOLD_FILE = EXPERIMENTS_DIR / "threshold.json"

# Defaults, stated so they can be argued with.
#
# FN = 40,000 (INR): an order-of-magnitude stand-in for what a candidate loses to a
# job scam — a "registration fee", a "training deposit", or documents used for
# identity theft. It is not a measured figure and nothing here depends on it being
# exactly right; `sweep_costs` exists precisely because it is a guess.
#
# FP = 200 (INR): a few minutes of a reviewer's time at a plausible loaded rate.
#
# The ratio, 200:1, is what actually determines the threshold. That ratio is the
# number to defend, not either absolute value.
DEFAULT_COST_FN = 40_000.0
DEFAULT_COST_FP = 200.0


@dataclass(frozen=True)
class ThresholdChoice:
    """A chosen threshold and the consequences of choosing it."""

    threshold: float
    rationale: str
    expected_cost: float
    caught: int
    missed: int
    false_alarms: int
    flagged: int
    precision: float
    recall: float

    # Set only by the capacity framing: the review budget, and how much of it went
    # unused because of tied scores. Slack is reported rather than silently absorbed.
    capacity: int | None = None
    slack: int | None = None

    def to_json(self) -> dict:
        return asdict(self)


def _outcomes(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> tuple[int, int, int]:
    flagged = y_score >= threshold
    tp = int(np.sum(flagged & y_true))
    fp = int(np.sum(flagged & ~y_true))
    fn = int(np.sum(~flagged & y_true))
    return tp, fp, fn


def _candidate_thresholds(y_score: np.ndarray) -> np.ndarray:
    """Every threshold that can change a decision, plus the endpoints.

    Scanning a fixed grid (0.00, 0.01, ...) would miss the optimum whenever scores
    cluster, which they do heavily under class weighting. Using the observed scores
    themselves makes the search exact rather than approximate.
    """
    unique = np.unique(y_score)
    return np.concatenate([[0.0], unique, [1.0 + 1e-9]])


def _describe(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float, rationale: str,
    cost_fn: float, cost_fp: float,
) -> ThresholdChoice:
    tp, fp, fn = _outcomes(y_true, y_score, threshold)
    flagged = tp + fp
    return ThresholdChoice(
        threshold=float(threshold),
        rationale=rationale,
        expected_cost=float(fn * cost_fn + fp * cost_fp),
        caught=tp,
        missed=fn,
        false_alarms=fp,
        flagged=flagged,
        precision=float(tp / flagged) if flagged else 0.0,
        recall=float(tp / (tp + fn)) if (tp + fn) else 0.0,
    )


def minimum_cost_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
) -> ThresholdChoice:
    """The threshold minimising `cost_fn * FN + cost_fp * FP`."""
    thresholds = _candidate_thresholds(y_score)
    costs = np.empty(len(thresholds))
    for i, t in enumerate(thresholds):
        _, fp, fn = _outcomes(y_true, y_score, t)
        costs[i] = fn * cost_fn + fp * cost_fp
    best = thresholds[int(np.argmin(costs))]
    return _describe(
        y_true, y_score, best,
        f"minimises expected cost at FN={cost_fn:,.0f} / FP={cost_fp:,.0f} "
        f"(ratio {cost_fn / cost_fp:.0f}:1)",
        cost_fn, cost_fp,
    )


def capacity_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    capacity: int,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
) -> ThresholdChoice:
    """The lowest threshold whose flag count fits within `capacity` reviews.

    Lowest, not highest: within a fixed review budget every unused slot is a fraud
    you could have caught for free. The constraint binds from above, so the correct
    move is to flag as much as the reviewer can actually process.

    **Capacity is often not filled exactly, and that is not a bug.** Tree ensembles
    produce heavily tied scores — on the portable/grouped test fold, 21 postings share
    the score 0.964643 — so the flag count can jump straight past the budget (34 -> 55
    at capacity 50). Filling the remaining slots would mean splitting a tie group
    arbitrarily, which is worse than leaving slots idle: two postings the model
    considers identical would get different treatment based on row order. `slack` in
    the returned choice records how many slots went unused.
    """
    thresholds = np.sort(_candidate_thresholds(y_score))
    chosen = thresholds[-1]
    for t in thresholds:
        tp, fp, _ = _outcomes(y_true, y_score, t)
        if tp + fp <= capacity:
            chosen = t
            break
    choice = _describe(
        y_true, y_score, chosen,
        f"largest flag set fitting {capacity} reviews",
        cost_fn, cost_fp,
    )
    return replace(choice, capacity=capacity, slack=capacity - choice.flagged)


def sweep_costs(
    y_true: np.ndarray,
    y_score: np.ndarray,
    ratios: tuple[float, ...] = (1, 5, 10, 25, 50, 100, 200, 500, 1000),
    cost_fp: float = DEFAULT_COST_FP,
) -> list[dict]:
    """How the chosen threshold moves as the FN:FP ratio changes.

    This is the honest answer to "where did 200:1 come from". If the threshold is
    stable across a wide band of ratios, the specific guess does not matter much; if
    it swings violently, the deck should say so.
    """
    rows = []
    for ratio in ratios:
        choice = minimum_cost_threshold(y_true, y_score, cost_fn=ratio * cost_fp, cost_fp=cost_fp)
        rows.append(
            {
                "fn_fp_ratio": float(ratio),
                "threshold": choice.threshold,
                "flagged": choice.flagged,
                "caught": choice.caught,
                "missed": choice.missed,
                "false_alarms": choice.false_alarms,
                "precision": round(choice.precision, 4),
                "recall": round(choice.recall, 4),
            }
        )
    return rows


def sweep_capacity(
    y_true: np.ndarray,
    y_score: np.ndarray,
    capacities: tuple[int, ...] = (10, 25, 50, 100, 200, 500),
) -> list[dict]:
    """Fraud caught per day as review headcount changes. The operator's curve."""
    rows = []
    for capacity in capacities:
        choice = capacity_threshold(y_true, y_score, capacity)
        rows.append(
            {
                "capacity": capacity,
                "threshold": choice.threshold,
                "flagged": choice.flagged,
                "caught": choice.caught,
                "missed": choice.missed,
                "precision": round(choice.precision, 4),
                "recall": round(choice.recall, 4),
            }
        )
    return rows


def analyse(
    y_true: np.ndarray,
    y_score: np.ndarray,
    capacity: int = 50,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
) -> dict:
    """Everything the threshold slide needs, in one dict."""
    all_flagged = _describe(
        y_true, y_score, 0.0, "flag everything (the do-nothing-clever baseline)",
        cost_fn, cost_fp,
    )
    none_flagged = _describe(
        y_true, y_score, 1.0 + 1e-9, "flag nothing (95% accurate, catches nothing)",
        cost_fn, cost_fp,
    )
    cost_choice = minimum_cost_threshold(y_true, y_score, cost_fn, cost_fp)
    capacity_choice = capacity_threshold(y_true, y_score, capacity, cost_fn, cost_fp)

    return {
        "costs": {"false_negative": cost_fn, "false_positive": cost_fp,
                  "ratio": cost_fn / cost_fp},
        "n": len(y_true),
        "n_fraud": int(y_true.sum()),
        "chosen": {
            "minimum_expected_cost": cost_choice.to_json(),
            "review_capacity": capacity_choice.to_json(),
        },
        "reference_points": {
            "flag_everything": all_flagged.to_json(),
            "flag_nothing": none_flagged.to_json(),
        },
        "cost_sweep": sweep_costs(y_true, y_score, cost_fp=cost_fp),
        "capacity_sweep": sweep_capacity(y_true, y_score),
    }


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Threshold analysis for a trained model.")
    parser.add_argument("--model", default="lightgbm")
    parser.add_argument("--regime", default="portable")
    parser.add_argument("--capacity", type=int, default=50)
    parser.add_argument("--cost-fn", type=float, default=DEFAULT_COST_FN)
    parser.add_argument("--cost-fp", type=float, default=DEFAULT_COST_FP)
    parser.add_argument("--sweep", action="store_true", help="print the sensitivity tables")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Imported here so `--help` does not pay for sklearn and the dataset.
    from veridyx import features as feat
    from veridyx.data import prepare
    from veridyx.evaluate import run_one

    dataset = prepare(seed=args.seed)
    features = feat.build(args.regime, dataset.postings)
    fold = dataset.frame["fold_grouped"].to_numpy()
    result = run_one(args.model, features, dataset.labels, fold, "grouped", args.seed)

    report = analyse(
        result.labels, result.scores, args.capacity, args.cost_fn, args.cost_fp
    )
    report["model"] = args.model
    report["regime"] = args.regime

    print(f"\n{args.model} / {args.regime} / grouped — test fold")
    print(f"  {report['n']:,} postings, {report['n_fraud']} fraudulent\n")
    for label, choice in report["chosen"].items():
        print(f"  {label}")
        print(f"    threshold {choice['threshold']:.4f}  — {choice['rationale']}")
        print(
            f"    flags {choice['flagged']:4d}   catches {choice['caught']:3d}   "
            f"misses {choice['missed']:3d}   P={choice['precision']:.3f} R={choice['recall']:.3f}"
        )

    if args.sweep:
        print("\ncost sensitivity (how much does the 200:1 guess matter?)")
        print(f"  {'FN:FP':>7s} {'thresh':>8s} {'flagged':>8s} {'caught':>7s} {'missed':>7s} {'P':>6s} {'R':>6s}")
        for row in report["cost_sweep"]:
            print(
                f"  {row['fn_fp_ratio']:7.0f} {row['threshold']:8.4f} {row['flagged']:8d} "
                f"{row['caught']:7d} {row['missed']:7d} {row['precision']:6.3f} {row['recall']:6.3f}"
            )

        print("\nreview capacity (what a real operator can act on)")
        print(f"  {'cap/day':>8s} {'thresh':>8s} {'flagged':>8s} {'caught':>7s} {'P':>6s} {'R':>6s}")
        for row in report["capacity_sweep"]:
            print(
                f"  {row['capacity']:8d} {row['threshold']:8.4f} {row['flagged']:8d} "
                f"{row['caught']:7d} {row['precision']:6.3f} {row['recall']:6.3f}"
            )

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    THRESHOLD_FILE.write_text(json.dumps(report, indent=2) + "\n")
    log.info("wrote %s", THRESHOLD_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
