"""The human review step: an explained flag sheet, and a record of what was decided.

This is what separates "human in the loop" as a claim from "human in the loop" as an
artifact. Two pieces:

**A self-contained HTML sheet.** Every posting a reviewer is asked to judge, with its
score, its SHAP attribution, and enough of the text to make a call — no server, no
build step, no external assets. It opens from disk and it will still open in a year.

**A verdict log.** `experiments/verdicts.jsonl`, append-only, one `ReviewVerdict` per
line, recording the model version and score *at the time of review*. A later retrain
must not be able to rewrite what a reviewer was actually shown.

**On reviewing postings that were not flagged.** The sheet takes the top N by score
regardless of threshold. On the quantyx feed nothing crosses the operating threshold
at all, so a strict flags-only sheet would be empty and there would be nothing to
review. "We examined the 25 highest-scoring live postings and confirmed each one" is
a real finding; an empty page is not.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from veridyx.data import ROOT
from veridyx.schema import Contribution, ReviewVerdict, ScoreRequest

log = logging.getLogger(__name__)

EXPERIMENTS_DIR = ROOT / "experiments"
REPORTS_DIR = ROOT / "reports"
VERDICTS_FILE = EXPERIMENTS_DIR / "verdicts.jsonl"

# Excerpt length. Long enough to judge a posting, short enough that a reviewer will
# actually read every card rather than skimming the first three.
EXCERPT_CHARS = 900

_CSS = """
:root { color-scheme: light dark;
  --surface:#fcfcfb; --card:#ffffff; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --blue:#2a78d6; --orange:#eb6834; }
@media (prefers-color-scheme: dark) { :root {
  --surface:#1a1a19; --card:#232322; --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --blue:#3987e5; --orange:#d95926; } }
* { box-sizing: border-box; }
body { margin:0; padding:32px 20px 64px; background:var(--surface); color:var(--ink);
  font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif; }
.wrap { max-width: 940px; margin: 0 auto; }
h1 { font-size:26px; margin:0 0 6px; letter-spacing:-0.01em; }
.sub { color:var(--muted); font-size:13.5px; margin-bottom:28px; }
.banner { border-left:3px solid var(--orange); background:color-mix(in srgb,var(--orange) 8%,transparent);
  padding:12px 16px; border-radius:0 6px 6px 0; margin-bottom:28px; font-size:14px; color:var(--ink2); }
.card { background:var(--card); border:1px solid var(--grid); border-radius:10px;
  padding:18px 20px; margin-bottom:16px; }
.head { display:flex; justify-content:space-between; align-items:baseline; gap:16px; }
.title { font-weight:650; font-size:16.5px; }
.score { font-variant-numeric:tabular-nums; font-weight:650; color:var(--blue); white-space:nowrap; }
.meta { color:var(--muted); font-size:13px; margin:4px 0 12px; }
.meta a { color:var(--muted); }
.contribs { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:12px; }
.chip { font-size:12.5px; padding:3px 9px; border-radius:999px; border:1px solid var(--grid);
  font-variant-numeric:tabular-nums; }
.chip.up { color:var(--orange); border-color:color-mix(in srgb,var(--orange) 40%,var(--grid)); }
.chip.down { color:var(--blue); border-color:color-mix(in srgb,var(--blue) 40%,var(--grid)); }
.excerpt { color:var(--ink2); font-size:13.5px; white-space:pre-wrap; border-top:1px solid var(--grid);
  padding-top:12px; margin:0; max-height:11em; overflow:auto; }
.cmd { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12.5px;
  color:var(--muted); margin-top:12px; }
table { border-collapse:collapse; width:100%; font-size:13.5px; }
th,td { text-align:left; padding:6px 10px; border-bottom:1px solid var(--grid); }
th { color:var(--muted); font-weight:500; }
"""


def _chip(contribution: Contribution) -> str:
    direction = "up" if contribution.value > 0 else "down"
    arrow = "▲" if contribution.value > 0 else "▼"
    return (
        f'<span class="chip {direction}">{arrow} {html.escape(contribution.feature)} '
        f"{contribution.value:+.2f}</span>"
    )


def render_sheet(
    requests: list[ScoreRequest],
    scores: np.ndarray,
    contributions: list[list[Contribution]],
    *,
    threshold: float,
    model_version: str,
    context: str = "",
    path: Path | None = None,
) -> Path:
    """Write a self-contained review sheet. Returns its path."""
    path = path or REPORTS_DIR / "review_sheet.html"
    path.parent.mkdir(parents=True, exist_ok=True)

    n_over = int((scores >= threshold).sum())
    cards = []
    for request, score, contribs in zip(requests, scores, contributions, strict=True):
        excerpt = (request.description or "")[:EXCERPT_CHARS]
        if request.description and len(request.description) > EXCERPT_CHARS:
            excerpt += " …"
        link = (
            f' · <a href="{html.escape(request.url)}">source</a>' if request.url else ""
        )
        cards.append(
            f"""<div class="card">
  <div class="head">
    <span class="title">{html.escape(request.title)}</span>
    <span class="score">{score:.4f}</span>
  </div>
  <div class="meta">{html.escape(request.company or "unknown company")}
    · {html.escape(request.location or "no location")}
    · <code>{html.escape(request.source_id or "-")}</code>{link}</div>
  <div class="contribs">{"".join(_chip(c) for c in contribs)}</div>
  <p class="excerpt">{html.escape(excerpt) or "<no description>"}</p>
  <div class="cmd">python -m veridyx.review --record {html.escape(request.source_id or "")} \
--decision fraud|legitimate|unclear</div>
</div>"""
        )

    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    banner = (
        f"<div class='banner'>{html.escape(context)}</div>" if context else ""
    )
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Veridyx — review sheet</title><style>{_CSS}</style></head>
<body><div class="wrap">
<h1>Review sheet</h1>
<div class="sub">{len(requests)} postings · model <code>{html.escape(model_version)}</code>
 · operating threshold {threshold:.4f} · {n_over} above it · generated {generated}</div>
{banner}
{"".join(cards)}
</div></body></html>
"""
    path.write_text(doc, encoding="utf-8")
    log.info("wrote %s", path)
    return path


def record(verdict: ReviewVerdict, path: Path | None = None) -> Path:
    """Append one human decision. Append-only by design — history is the point."""
    path = path or VERDICTS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(verdict.model_dump_json() + "\n")
    log.info("recorded %s -> %s", verdict.source_id, verdict.decision)
    return path


def load_verdicts(path: Path | None = None) -> list[ReviewVerdict]:
    path = path or VERDICTS_FILE
    if not path.exists():
        return []
    return [
        ReviewVerdict(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the review sheet, or record a verdict.")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--capacity", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quantyx-root", type=Path, default=None)
    parser.add_argument("--record", metavar="SOURCE_ID", help="record a decision instead")
    parser.add_argument("--decision", choices=["fraud", "legitimate", "unclear"])
    parser.add_argument("--note", default=None)
    parser.add_argument("--score", type=float, default=0.0)
    parser.add_argument("--model-version", default="unknown")
    parser.add_argument("--summary", action="store_true", help="print recorded verdicts")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.summary:
        verdicts = load_verdicts()
        if not verdicts:
            print("no verdicts recorded yet")
            return 0
        counts: dict[str, int] = {}
        for v in verdicts:
            counts[v.decision] = counts.get(v.decision, 0) + 1
        print(f"{len(verdicts)} verdicts recorded")
        for decision, count in sorted(counts.items()):
            print(f"  {decision:12s} {count}")
        return 0

    if args.record:
        if not args.decision:
            parser.error("--record requires --decision")
        record(
            ReviewVerdict(
                source_id=args.record,
                decision=args.decision,
                model_score=args.score,
                model_version=args.model_version,
                note=args.note,
            )
        )
        return 0

    from veridyx import features as feat
    from veridyx.adapters.quantyx import load_feed, score_feed
    from veridyx.data import prepare
    from veridyx.evaluate import _subset
    from veridyx.explain import explain
    from veridyx.features import portable_features
    from veridyx.models.gbm import GradientBoosting
    from veridyx.splits import TRAIN, VAL
    from veridyx.threshold import capacity_threshold

    dataset = prepare(seed=args.seed)
    features = feat.build("portable", dataset.postings)
    fold = dataset.frame["fold_grouped"].to_numpy()
    model = GradientBoosting(seed=args.seed).fit(
        _subset(features, fold == TRAIN), dataset.labels[fold == TRAIN]
    )
    operating = capacity_threshold(
        dataset.labels[fold == VAL],
        model.predict_proba(_subset(features, fold == VAL)),
        args.capacity,
    )

    postings = load_feed(args.quantyx_root)
    scores, _ = score_feed(postings, model, operating.threshold)
    order = np.argsort(scores)[::-1][: args.top]

    requests = [postings[i].request for i in order]
    subset = portable_features(requests)
    contributions = explain(model, subset, k=6)

    n_over = int((scores >= operating.threshold).sum())
    context = (
        f"Nothing in this feed crosses the operating threshold of "
        f"{operating.threshold:.4f} — these are the {args.top} highest-scoring postings "
        f"out of {len(postings):,}, shown so the model's ranking can be judged even "
        "when it flags nothing. Check experiments/drift.json before reading a quiet "
        "model as a clean feed."
        if n_over == 0
        else f"{n_over} of {len(postings):,} postings cross the operating threshold."
    )

    path = render_sheet(
        requests, scores[order], contributions,
        threshold=operating.threshold,
        model_version=model.version,
        context=context,
    )
    print(f"\nreview sheet: {path}")
    print(f"  {len(postings):,} live postings, {n_over} above threshold, showing top {args.top}")
    print(f"  open it with:  open {path}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
