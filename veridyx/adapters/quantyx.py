"""Read the quantyx live feed and score it with the deployed model.

quantyx (github.com/yugvyas/quantyx) is a sibling project by the same author: a
self-updating pipeline that pulls India DS/AI job postings from Adzuna, Greenhouse,
Lever and Ashby every morning and commits the raw results to its own repository as
gzipped JSONL. That commit-the-dataset design is what makes this adapter possible
without any network access, API key, or coupling between the two codebases — Veridyx
reads files, and the files are already on disk.

**What this evaluation can and cannot show.** These postings are unlabelled, current,
and drawn from a distribution nobody trained on. They are also, overwhelmingly,
legitimate: Greenhouse/Lever/Ashby are per-company ATS boards for real employers and
Adzuna is a mainstream aggregator, so the true fraud rate here is near zero. A high
flag count would therefore be evidence of *false positives under domain shift*, not
of fraud discovered. That is the finding this module is built to measure honestly,
and the reason `drift.py` runs alongside it rather than afterwards.

The mapping is deliberately narrow. quantyx's `Posting` carries compensation fields,
seniority, an India flag and a lifespan model, none of which EMSCAD has — so none of
them may become features here, however tempting. Only what `ScoreRequest` already
declares crosses the boundary.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from veridyx.data import ROOT
from veridyx.schema import ScoreRequest

log = logging.getLogger(__name__)

# quantyx sits beside Veridyx by default. Overridable because a clean clone of this
# repository will not have it, and the failure must be a clear message rather than an
# empty result set that silently reports "0 postings scored".
DEFAULT_QUANTYX_ROOT = ROOT.parent / "quantyx"
EXPERIMENTS_DIR = ROOT / "experiments"
LIVE_FEED_FILE = EXPERIMENTS_DIR / "live_feed.json"


class FeedNotFound(RuntimeError):
    """Raised when the quantyx dataset cannot be located."""


@dataclass(frozen=True)
class LivePosting:
    """A quantyx posting, plus the identity needed to review it later."""

    request: ScoreRequest
    source: str
    posted_date: str | None
    fetched_at: str | None


def _iter_records(postings_dir: Path):
    """Yield every posting record from quantyx's gzipped JSONL partitions."""
    files = sorted(postings_dir.glob("*/*.jsonl.gz"))
    if not files:
        raise FeedNotFound(
            f"no posting partitions under {postings_dir}. Expected quantyx's "
            "data/postings/<date>/<source>.jsonl.gz layout."
        )
    for path in files:
        with gzip.open(path, "rt") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)


def load_feed(quantyx_root: Path | None = None) -> list[LivePosting]:
    """Read and normalise every posting quantyx has stored.

    quantyx writes a full record once, on first sight, so this is naturally
    deduplicated — there is no need to collapse repeat observations.
    """
    root = Path(quantyx_root or DEFAULT_QUANTYX_ROOT)
    postings_dir = root / "data" / "postings"
    if not postings_dir.exists():
        raise FeedNotFound(
            f"{postings_dir} does not exist.\n"
            "Veridyx expects the quantyx repository beside it. Clone it with:\n"
            "  git clone https://github.com/yugvyas/quantyx\n"
            "or pass --quantyx-root explicitly."
        )

    out: list[LivePosting] = []
    skipped = 0
    for record in _iter_records(postings_dir):
        try:
            request = ScoreRequest(
                title=record["title"],
                description=record.get("description"),
                location=record.get("location"),
                is_remote=bool(record.get("is_remote", False)),
                # quantyx carries structured compensation; EMSCAD only ever had a
                # free-text range. Presence is the one thing both can express, so
                # presence is all that crosses.
                has_salary=record.get("comp_min") is not None
                or record.get("comp_max") is not None,
                company=record.get("company"),
                url=record.get("url"),
                source_id=f"{record.get('source')}:{record.get('posting_id')}",
            )
        except Exception as exc:
            skipped += 1
            log.debug("skipped %s: %s", record.get("posting_id"), exc)
            continue
        out.append(
            LivePosting(
                request=request,
                source=str(record.get("source", "unknown")),
                posted_date=record.get("posted_date"),
                fetched_at=record.get("fetched_at"),
            )
        )

    if skipped:
        log.warning("skipped %d malformed records", skipped)
    log.info("loaded %d live postings from %s", len(out), postings_dir)
    return out


def false_positive_drivers(model, postings: list[LivePosting], order, k: int = 6) -> dict:
    """Which features push the highest-scoring *live* postings upward, and why.

    On a feed with no known fraud, every high score is a false positive in waiting, so
    aggregating their SHAP drivers is the error analysis. Doing it here rather than by
    eye means the finding is regenerated with the data instead of being frozen in a
    slide the moment the feed changes.

    The mechanism this surfaced on the first run is worth stating, because it is a
    feature bug rather than a modelling one: the model learned the bigram "work from"
    from "work from home", and a legitimate posting advertising "5 Days Work From
    Office" trips it. Two other patterns showed up alongside it — fintech vocabulary
    ("money", "income") colliding with scam vocabulary, and Adzuna's contact-redaction
    artifacts ("hidden email orhidden mobileto") inflating punctuation_ratio.
    """
    from veridyx.explain import explain
    from veridyx.features import portable_features

    subset = portable_features([postings[i].request for i in order])
    tally: dict[str, dict] = {}
    for contributions in explain(model, subset, k=k):
        for contribution in contributions:
            if contribution.value <= 0:
                continue
            entry = tally.setdefault(
                contribution.feature, {"postings": 0, "total_push": 0.0}
            )
            entry["postings"] += 1
            entry["total_push"] += contribution.value

    ranked = sorted(tally.items(), key=lambda kv: -kv[1]["total_push"])
    return {
        "n_examined": len(order),
        "drivers": [
            {
                "feature": name,
                "postings_affected": stats["postings"],
                "total_push_toward_fraud": round(stats["total_push"], 4),
            }
            for name, stats in ranked[:15]
        ],
    }


def score_feed(
    postings: list[LivePosting], model, threshold: float
) -> tuple[np.ndarray, list]:
    """Score the feed with an already-trained PORTABLE model.

    Takes a fitted model rather than training one, so the thing evaluated against the
    live feed is provably the same object evaluated on the benchmark.
    """
    from veridyx.features import PORTABLE, portable_features

    if model.regime != PORTABLE:
        raise ValueError(
            f"the live feed can only be scored by a {PORTABLE!r} model; this one is "
            f"{model.regime!r}. A FULL model depends on columns quantyx does not have."
        )
    features = portable_features([p.request for p in postings])
    return model.predict_proba(features), features


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score the quantyx live feed.")
    parser.add_argument("--quantyx-root", type=Path, default=None)
    parser.add_argument("--top", type=int, default=25, help="how many top flags to show")
    parser.add_argument("--capacity", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from veridyx import features as feat
    from veridyx.data import prepare
    from veridyx.evaluate import _subset
    from veridyx.models.gbm import GradientBoosting
    from veridyx.splits import TRAIN, VAL
    from veridyx.threshold import capacity_threshold

    postings = load_feed(args.quantyx_root)

    # Train on EMSCAD exactly as the benchmark run does, then choose the operating
    # threshold on the EMSCAD validation fold. Choosing it on the live feed would be
    # circular: there are no labels there to choose against.
    dataset = prepare(seed=args.seed)
    features = feat.build("portable", dataset.postings)
    fold = dataset.frame["fold_grouped"].to_numpy()
    model = GradientBoosting(seed=args.seed).fit(
        _subset(features, fold == TRAIN), dataset.labels[fold == TRAIN]
    )
    val_scores = model.predict_proba(_subset(features, fold == VAL))
    operating = capacity_threshold(
        dataset.labels[fold == VAL], val_scores, args.capacity
    )

    scores, _ = score_feed(postings, model, operating.threshold)
    flagged = scores >= operating.threshold

    report = {
        "n_postings": len(postings),
        "threshold": operating.threshold,
        "threshold_rationale": operating.rationale,
        "n_flagged": int(flagged.sum()),
        "flag_rate": float(flagged.mean()),
        "score_quantiles": {
            q: float(np.quantile(scores, q / 100)) for q in (50, 75, 90, 95, 99)
        },
        "by_source": {},
        "model_version": model.version,
    }
    for source in sorted({p.source for p in postings}):
        mask = np.array([p.source == source for p in postings])
        report["by_source"][source] = {
            "n": int(mask.sum()),
            "n_flagged": int(flagged[mask].sum()),
            "mean_score": float(scores[mask].mean()),
        }

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    LIVE_FEED_FILE.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\nquantyx live feed — {report['n_postings']:,} postings")
    print(f"  model     {report['model_version']}")
    print(f"  threshold {report['threshold']:.4f}  ({report['threshold_rationale']})")
    print(f"  flagged   {report['n_flagged']} ({report['flag_rate']:.2%})")
    print("\n  score quantiles")
    for q, value in report["score_quantiles"].items():
        print(f"    p{q:<3d} {value:.4f}")
    print("\n  by source")
    for source, stats in report["by_source"].items():
        print(
            f"    {source:12s} n={stats['n']:5,d}  flagged={stats['n_flagged']:3d}  "
            f"mean score={stats['mean_score']:.4f}"
        )

    order = np.argsort(scores)[::-1][: args.top]
    report["false_positive_drivers"] = false_positive_drivers(model, postings, order)
    LIVE_FEED_FILE.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\n  what pushes the top {args.top} upward (error analysis)")
    for driver in report["false_positive_drivers"]["drivers"][:8]:
        print(
            f"    {driver['total_push_toward_fraud']:+7.2f}  {driver['feature'][:30]:30s} "
            f"({driver['postings_affected']} postings)"
        )

    print(f"\n  top {args.top} by score")
    for rank, i in enumerate(order, 1):
        posting = postings[i]
        print(
            f"    {rank:2d}. {scores[i]:.4f}  {posting.request.title[:52]:52s} "
            f"| {(posting.request.company or '?')[:22]}"
        )

    log.info("wrote %s", LIVE_FEED_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
