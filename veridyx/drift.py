"""Drift monitoring: is the live feed the kind of data this model was trained on?

A fraud model deployed against a feed it has never seen can fail in two directions,
and only one of them is visible from the flag count:

* **It fires constantly.** Obvious, noisy, self-reporting.
* **It goes quiet.** Also a failure — a model whose scores have collapsed toward zero
  because the input no longer resembles training data looks exactly like a model
  finding nothing wrong. The flag count cannot tell those apart.

This module measures the second case. Three complementary signals:

**PSI (Population Stability Index)** on the score distribution — the standard measure
of how far a deployed score distribution has moved from its reference. The
conventional reading is < 0.1 stable, 0.1 to 0.25 moderate, > 0.25 significant.

**KS statistic** on the same two distributions, as a distribution-shape check that
does not depend on the bin edges PSI happens to choose.

**Out-of-vocabulary rate** — the share of live tokens the trained vectoriser has never
seen. This is the one that explains *why* the scores moved: a model whose vocabulary
does not cover the new text cannot score it meaningfully, however confident it looks.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from veridyx.data import ROOT
from veridyx.text import tokenize

log = logging.getLogger(__name__)

EXPERIMENTS_DIR = ROOT / "experiments"
DRIFT_FILE = EXPERIMENTS_DIR / "drift.json"

# Conventional PSI bands. Stated here rather than inline so the thresholds the report
# is judged against are visible in one place.
PSI_STABLE = 0.10
PSI_SIGNIFICANT = 0.25


@dataclass(frozen=True)
class DriftReport:
    psi: float
    psi_verdict: str
    ks_statistic: float
    oov_rate: float
    reference_mean: float
    live_mean: float
    reference_n: int
    live_n: int
    notes: list[str]

    def to_json(self) -> dict:
        return asdict(self)


def population_stability_index(
    reference: np.ndarray, live: np.ndarray, bins: int = 10
) -> float:
    """PSI between two score distributions.

    Bin edges come from the *reference* quantiles, not from a uniform 0-1 grid: fraud
    scores are extremely skewed (most mass near zero), and uniform bins would put
    almost everything in the first bucket and report stability no matter what happened.

    Empty bins are floored rather than dropped. A live bin that is genuinely empty is
    real evidence of drift, and dropping it would hide exactly the signal being
    measured; the floor keeps the logarithm finite without erasing it.
    """
    if reference.size == 0 or live.size == 0:
        return 0.0

    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(reference, quantiles))
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    live_counts, _ = np.histogram(live, bins=edges)

    floor = 1e-6
    ref_share = np.maximum(ref_counts / reference.size, floor)
    live_share = np.maximum(live_counts / live.size, floor)
    return float(np.sum((live_share - ref_share) * np.log(live_share / ref_share)))


def ks_statistic(reference: np.ndarray, live: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic: the largest CDF gap."""
    from scipy.stats import ks_2samp

    if reference.size == 0 or live.size == 0:
        return 0.0
    return float(ks_2samp(reference, live).statistic)


def oov_rate(vectorizer, texts: list[str]) -> float:
    """Share of live tokens absent from the trained vocabulary.

    Token-weighted rather than type-weighted: a rare unknown word appearing once
    matters far less than a common one appearing in every posting, and the
    type-weighted figure would overstate drift by counting both the same.
    """
    vocabulary = set(vectorizer.vocabulary_)
    total = 0
    unknown = 0
    for text in texts:
        for token in tokenize(text):
            total += 1
            if token not in vocabulary:
                unknown += 1
    return float(unknown / total) if total else 0.0


def _verdict(psi: float) -> str:
    if psi < PSI_STABLE:
        return "stable"
    if psi < PSI_SIGNIFICANT:
        return "moderate shift"
    return "significant shift"


def analyse(
    reference_scores: np.ndarray,
    live_scores: np.ndarray,
    vectorizer=None,
    live_texts: list[str] | None = None,
) -> DriftReport:
    """Compare a live score distribution against its training-time reference."""
    psi = population_stability_index(reference_scores, live_scores)
    notes: list[str] = []

    rate = oov_rate(vectorizer, live_texts) if vectorizer and live_texts else 0.0

    if live_scores.mean() < reference_scores.mean() * 0.5:
        notes.append(
            "Live scores average less than half the reference. A quiet model is not "
            "the same as a clean feed — check the OOV rate before concluding the "
            "absence of flags means the absence of fraud."
        )
    if rate > 0.25:
        notes.append(
            f"{rate:.1%} of live tokens are outside the trained vocabulary. The model "
            "is scoring text it largely cannot read."
        )
    if psi >= PSI_SIGNIFICANT:
        notes.append(
            f"PSI {psi:.3f} exceeds the conventional {PSI_SIGNIFICANT} threshold for "
            "significant distribution shift; scores are not comparable to the "
            "benchmark and the operating threshold should be re-derived."
        )

    return DriftReport(
        psi=round(psi, 4),
        psi_verdict=_verdict(psi),
        ks_statistic=round(ks_statistic(reference_scores, live_scores), 4),
        oov_rate=round(rate, 4),
        reference_mean=round(float(reference_scores.mean()), 6),
        live_mean=round(float(live_scores.mean()), 6),
        reference_n=int(reference_scores.size),
        live_n=int(live_scores.size),
        notes=notes,
    )


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare the quantyx live feed against the EMSCAD reference."
    )
    parser.add_argument("--quantyx-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from veridyx import features as feat
    from veridyx.adapters.quantyx import load_feed, score_feed
    from veridyx.data import prepare
    from veridyx.evaluate import _subset
    from veridyx.features import portable_features
    from veridyx.models.gbm import GradientBoosting
    from veridyx.splits import TEST, TRAIN

    dataset = prepare(seed=args.seed)
    features = feat.build("portable", dataset.postings)
    fold = dataset.frame["fold_grouped"].to_numpy()
    model = GradientBoosting(seed=args.seed).fit(
        _subset(features, fold == TRAIN), dataset.labels[fold == TRAIN]
    )
    # The reference is the held-out test fold, not the training fold: training scores
    # are optimistically peaked and would make any live feed look like drift.
    reference = model.predict_proba(_subset(features, fold == TEST))

    postings = load_feed(args.quantyx_root)
    live, _ = score_feed(postings, model, 0.5)
    live_texts = portable_features([p.request for p in postings]).texts

    report = analyse(reference, live, model.vectorizer, live_texts)

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    DRIFT_FILE.write_text(json.dumps(report.to_json(), indent=2) + "\n")

    print("\ndrift: EMSCAD test fold  ->  quantyx live feed")
    print(f"  PSI            {report.psi:.4f}   ({report.psi_verdict})")
    print(f"  KS statistic   {report.ks_statistic:.4f}")
    print(f"  OOV rate       {report.oov_rate:.2%}")
    print(f"  mean score     {report.reference_mean:.6f} -> {report.live_mean:.6f}")
    print(f"  n              {report.reference_n:,} -> {report.live_n:,}")
    if report.notes:
        print("\n  notes")
        for note in report.notes:
            print(f"    - {note}")
    log.info("wrote %s", DRIFT_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
