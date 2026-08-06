"""Regenerate the README's results block from committed experiment output.

Mirrors quantyx's `<!-- STATS:START -->` pattern, and for the same reason: a number
typed into a README is a number that goes stale silently. Everything between the
markers is rewritten from `experiments/*.json`, so the README cannot drift from the
runs that produced it.

Run after any experiment:  .venv/bin/python -m veridyx.stats
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from veridyx.data import ROOT

log = logging.getLogger(__name__)

README = ROOT / "README.md"
EXPERIMENTS_DIR = ROOT / "experiments"
START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"


def _load(name: str) -> dict | None:
    path = EXPERIMENTS_DIR / name
    return json.loads(path.read_text()) if path.exists() else None


def render() -> str:
    """Build the markdown block. Missing experiments degrade to a note, not a crash."""
    lines: list[str] = []

    dataset = _load("dataset_stats.json")
    if dataset:
        dup = dataset["near_duplicates"]
        lines += [
            "### The dataset, after de-duplication",
            "",
            f"**{dup['n_documents']:,}** postings · **{dup['n_clusters']:,}** near-duplicate "
            f"clusters · **{dup['duplicate_share_fraudulent']:.1%}** of fraudulent postings "
            "have a twin",
            "",
            "| | share sitting in a near-duplicate cluster |",
            "| --- | --- |",
            f"| All postings | {dup['duplicate_share_overall']:.1%} |",
            f"| Fraudulent | **{dup['duplicate_share_fraudulent']:.1%}** "
            f"({dup['n_fraudulent_with_duplicate']} of {dup['n_fraudulent']}) |",
            f"| Legitimate | {dup['duplicate_share_legitimate']:.1%} |",
            "",
        ]

    results = _load("results.json")
    if results:
        lines += [
            "### Results",
            "",
            "Grouped split, held-out test fold. Thresholds chosen on validation.",
            "",
            "| model | regime | PR-AUC | F1 | precision | recall |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        grouped = [r for r in results["runs"] if r["split_kind"] == "grouped"]
        for run in sorted(grouped, key=lambda r: (r["regime"], -r["test"]["pr_auc"])):
            t = run["test"]
            lines.append(
                f"| {run['model']} | {run['regime']} | {t['pr_auc']:.4f} | {t['f1']:.4f} "
                f"| {t['precision']:.4f} | {t['recall']:.4f} |"
            )
        lines.append("")

        if results.get("regime_gap"):
            lines += [
                "**What the benchmark-only columns are worth.** PR-AUC lost when the "
                "fields no live feed provides are removed:",
                "",
                "| model | FULL | PORTABLE | difference |",
                "| --- | ---: | ---: | ---: |",
            ]
            for row in results["regime_gap"]:
                lines.append(
                    f"| {row['model']} | {row['full_pr_auc']:.4f} | "
                    f"{row['portable_pr_auc']:.4f} | **+{row['benchmark_only_advantage']:.4f}** |"
                )
            lines.append("")

        if results.get("leakage_delta"):
            lines += [
                "**What a random split is worth.** PR-AUC gained by letting "
                "near-duplicates leak across folds:",
                "",
                "| model | regime | grouped | naive | inflation |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
            for row in results["leakage_delta"]:
                lines.append(
                    f"| {row['model']} | {row['regime']} | {row['grouped_pr_auc']:.4f} | "
                    f"{row['naive_pr_auc']:.4f} | **+{row['pr_auc_inflation']:.4f}** |"
                )
            lines.append("")

    live = _load("live_feed.json")
    drift = _load("drift.json")
    if live and drift:
        lines += [
            "### Against a live feed",
            "",
            f"Scored **{live['n_postings']:,}** current postings from "
            f"[quantyx](https://github.com/yugvyas/quantyx) at the deployed threshold of "
            f"{live['threshold']:.4f}.",
            "",
            f"- **{live['n_flagged']} flagged** ({live['flag_rate']:.2%}); "
            f"99th-percentile score {live['score_quantiles']['99']:.4f}",
            f"- Drift **PSI {drift['psi']:.4f}** ({drift['psi_verdict']}), "
            f"KS {drift['ks_statistic']:.4f}, OOV {drift['oov_rate']:.2%}",
            f"- Mean score {drift['reference_mean']:.4f} → {drift['live_mean']:.4f}",
            "",
            "Zero flags is not the same as a clean feed. The score distribution has "
            "shifted significantly, so the absence of flags is at least partly the "
            "model going quiet on data it was not trained for — which is exactly what "
            "the drift monitor exists to distinguish.",
            "",
        ]

    return "\n".join(lines).rstrip() or "_No experiments have been run yet._"


def update(path: Path | None = None) -> Path:
    path = path or README
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    text = path.read_text()
    if START not in text or END not in text:
        raise ValueError(f"{path} is missing the {START} / {END} markers")

    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    path.write_text(f"{head}{START}\n\n{render()}\n\n{END}{tail}")
    log.info("updated %s", path)
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    update()
    sys.exit(0)
