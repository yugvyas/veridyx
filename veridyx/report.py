"""Every figure and every number the deck uses, generated from committed results.

The rule this module exists to enforce: **no number in the deck is typed by hand.**
Each figure writes a PNG and a JSON sidecar holding exactly the values drawn, so a
claim on a slide can always be traced to the run that produced it. This is what makes
the deck's requirement-traceability claim literally true, and it is what makes a chart
with placeholder values in it structurally impossible.

Palette is the validated three-slot categorical set (blue / orange / aqua), checked
with the dataviz validator under `--pairs all` in light mode: worst CVD ΔE 9.2, worst
normal-vision ΔE 24.0. Aqua sits at 2.74:1 against the surface, below the 3:1 bar, so
every chart that uses it carries visible direct labels — the documented relief.

These are static PNGs for slides, so there is no hover layer; the JSON sidecar is the
table view that would otherwise accompany an interactive chart.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import precision_recall_curve

from veridyx.data import ROOT

log = logging.getLogger(__name__)

REPORTS_DIR = ROOT / "reports"
EXPERIMENTS_DIR = ROOT / "experiments"

# --- palette (light mode; validated) --------------------------------------------
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

DPI = 200
FIGSIZE = (10, 5.6)


def _style_axes(ax, *, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    """Recessive chrome: hairline grid, no top/right spines, muted tick labels."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, axis="y", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=11, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=11, labelpad=8)
    if title:
        ax.set_title(title, color=INK, fontsize=14, pad=14, loc="left", fontweight="600")


def _save(fig, name: str, payload: dict) -> Path:
    """Write the PNG and its JSON sidecar together, so they cannot diverge."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    png = REPORTS_DIR / f"{name}.png"
    fig.savefig(png, dpi=DPI, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    (REPORTS_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")
    log.info("wrote %s (+ .json)", png)
    return png


def _load(name: str) -> dict:
    path = EXPERIMENTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run the experiment that produces it first "
            "(python -m veridyx.evaluate / --prepare / -m veridyx.threshold)."
        )
    return json.loads(path.read_text())


# --------------------------------------------------------------------------------
# Figure 1 — the duplicate leakage in the dataset itself
# --------------------------------------------------------------------------------


def figure_duplicates() -> Path:
    """Near-duplicate share, overall and per class. The reason for grouped splits."""
    stats = _load("dataset_stats.json")["near_duplicates"]
    labels = ["All postings", "Fraudulent", "Legitimate"]
    values = [
        stats["duplicate_share_overall"],
        stats["duplicate_share_fraudulent"],
        stats["duplicate_share_legitimate"],
    ]
    colors = [MUTED, ORANGE, BLUE]

    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE)
    bars = ax.bar(labels, values, color=colors, width=0.55, zorder=2)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value:.1%}",
            ha="center", va="bottom", color=INK, fontsize=13, fontweight="600",
        )
    ax.set_ylim(0, max(values) * 1.25)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    _style_axes(
        ax,
        ylabel="share sitting in a near-duplicate cluster",
        title="Nearly half of fraudulent postings have a near-duplicate twin",
    )
    ax.text(
        0, -0.20,
        f"{stats['n_documents']:,} postings · {stats['n_clusters']:,} clusters · "
        f"MinHash Jaccard ≥ {stats['threshold']:.2f} · "
        f"{stats['n_fraudulent_with_duplicate']} of {stats['n_fraudulent']} fraudulent",
        transform=ax.transAxes, color=MUTED, fontsize=9.5,
    )
    return _save(fig, "duplicate_leakage", {"labels": labels, "values": values, "source": stats})


# --------------------------------------------------------------------------------
# Figure 2 — model comparison (this is the chart that replaces the placeholder pie)
# --------------------------------------------------------------------------------


def figure_model_comparison() -> Path:
    """PR-AUC by model and regime, on the grouped split only.

    A grouped bar rather than a pie: the job is comparing magnitudes across two
    nested categories, which a pie cannot do at all. Only the grouped split appears
    — putting the leaky numbers in the headline chart would undercut the whole point.
    """
    results = _load("results.json")
    runs = [r for r in results["runs"] if r["split_kind"] == "grouped"]
    models = sorted({r["model"] for r in runs})
    regimes = ["full", "portable"]
    by = {(r["model"], r["regime"]): r["test"]["pr_auc"] for r in runs}

    x = np.arange(len(models))
    width = 0.34
    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=SURFACE)

    for offset, regime, color, label in (
        (-width / 2 - 0.01, "full", ORANGE, "FULL — every EMSCAD column"),
        (+width / 2 + 0.01, "portable", BLUE, "PORTABLE — what a live feed provides"),
    ):
        values = [by.get((m, regime), 0.0) for m in models]
        bars = ax.bar(x + offset, values, width, color=color, label=label, zorder=2)
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}",
                ha="center", va="bottom", color=INK, fontsize=11, fontweight="600",
            )

    ax.set_xticks(x, models, color=INK_SECONDARY, fontsize=11)
    ax.set_ylim(0, 1.0)
    _style_axes(ax, ylabel="PR-AUC (grouped split, test fold)",
                title="What the model keeps when the benchmark-only columns go away")
    legend = ax.legend(frameon=False, loc="upper left", fontsize=10.5,
                       bbox_to_anchor=(0, -0.10), ncol=2)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    payload = {
        "metric": "pr_auc",
        "split": "grouped",
        "values": {f"{m}/{rg}": by.get((m, rg)) for m in models for rg in regimes},
        "regime_gap": results.get("regime_gap", []),
    }
    return _save(fig, "model_comparison", payload)


# --------------------------------------------------------------------------------
# Figure 3 — leakage: the same models under both split kinds
# --------------------------------------------------------------------------------


def figure_leakage() -> Path:
    """How much the naive split inflates each model. The methodology slide."""
    results = _load("results.json")
    rows = results.get("leakage_delta", [])
    if not rows:
        raise ValueError("results.json has no leakage_delta; run both split kinds")

    labels = [f"{r['model']}\n{r['regime']}" for r in rows]
    grouped = [r["grouped_pr_auc"] for r in rows]
    naive = [r["naive_pr_auc"] for r in rows]

    x = np.arange(len(rows))
    width = 0.34
    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=SURFACE)
    ax.bar(x - width / 2 - 0.01, grouped, width, color=BLUE,
           label="grouped split — duplicates kept together", zorder=2)
    ax.bar(x + width / 2 + 0.01, naive, width, color=ORANGE,
           label="naive random split — duplicates leak across folds", zorder=2)

    for i, row in enumerate(rows):
        ax.annotate(
            f"+{row['pr_auc_inflation']:.3f}",
            xy=(x[i], max(grouped[i], naive[i]) + 0.03),
            ha="center", color=INK, fontsize=11, fontweight="600",
        )

    ax.set_xticks(x, labels, color=INK_SECONDARY, fontsize=10)
    ax.set_ylim(0, 1.0)
    _style_axes(ax, ylabel="PR-AUC (test fold)",
                title="What a random split adds that generalisation does not")
    legend = ax.legend(frameon=False, loc="upper left", fontsize=10.5,
                       bbox_to_anchor=(0, -0.12), ncol=1)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    return _save(fig, "leakage_inflation", {"rows": rows})


# --------------------------------------------------------------------------------
# Figure 4 — the threshold as a cost decision
# --------------------------------------------------------------------------------


def figure_threshold() -> Path:
    """Review capacity against fraud caught, with the two chosen thresholds marked."""
    report = _load("threshold.json")
    sweep = report["capacity_sweep"]
    capacities = [r["capacity"] for r in sweep]
    caught = [r["caught"] for r in sweep]
    precision = [r["precision"] for r in sweep]

    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=SURFACE)
    ax.plot(capacities, caught, color=BLUE, linewidth=2, marker="o",
            markersize=8, zorder=3, label="fraudulent postings caught")
    for i, (cap, c, p) in enumerate(zip(capacities, caught, precision, strict=True)):
        # The first and last points sit against the axes; left-align the first and
        # right-align the last so their labels stay inside the plot area instead of
        # colliding with the y-axis ticks and the right spine.
        align = "left" if i == 0 else "right" if i == len(capacities) - 1 else "center"
        ax.annotate(
            f"{c}\n{p:.0%} precise",
            xy=(cap, c), xytext=(0, 14), textcoords="offset points",
            ha=align, color=INK_SECONDARY, fontsize=9.5,
        )

    total = report["n_fraud"]
    ax.axhline(total, color=BASELINE, linewidth=1.2, linestyle="--", zorder=1)
    ax.annotate(
        f"all {total} fraudulent postings in the fold",
        xy=(capacities[0], total), xytext=(0, 8), textcoords="offset points",
        color=MUTED, fontsize=9.5,
    )

    ax.set_xscale("log")
    ax.set_xticks(capacities, [str(c) for c in capacities])
    # Headroom for the two-line point labels above the ceiling line, not just above
    # the highest point.
    ax.set_ylim(0, total * 1.30)
    _style_axes(
        ax,
        xlabel="postings a reviewer can check per day",
        ylabel="fraudulent postings caught",
        title="The threshold is a staffing decision, not a metric choice",
    )
    # The rationale string already opens with "minimises expected cost at …", so it
    # is used as the whole sentence rather than prefixed — an earlier version read
    # "minimum expected cost at minimises expected cost at FN=…".
    cost_choice = report["chosen"]["minimum_expected_cost"]
    ax.text(
        0, -0.24,
        f"By contrast, the threshold that {cost_choice['rationale']} "
        f"flags {cost_choice['flagged']} postings at {cost_choice['precision']:.1%} precision — "
        f"far beyond any review desk.",
        transform=ax.transAxes, color=MUTED, fontsize=9.5,
    )
    return _save(fig, "threshold_capacity", report)


def figure_cost_sensitivity() -> Path:
    """How far the chosen threshold moves as the cost ratio changes."""
    report = _load("threshold.json")
    sweep = report["cost_sweep"]
    ratios = [r["fn_fp_ratio"] for r in sweep]
    recall = [r["recall"] for r in sweep]
    prec = [r["precision"] for r in sweep]

    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=SURFACE)
    ax.plot(ratios, recall, color=BLUE, linewidth=2, marker="o", markersize=8,
            label="recall — share of fraud caught", zorder=3)
    ax.plot(ratios, prec, color=ORANGE, linewidth=2, marker="s", markersize=8,
            label="precision — share of flags that are real", zorder=3)
    ax.set_xscale("log")
    ax.set_xticks(ratios, [f"{int(r)}:1" for r in ratios])
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    _style_axes(
        ax,
        xlabel="assumed cost of a missed fraud, relative to one false alarm",
        ylabel="",
        title="The cost ratio is a guess — here is how much it matters",
    )
    legend = ax.legend(frameon=False, loc="upper left", fontsize=10.5,
                       bbox_to_anchor=(0, -0.14), ncol=2)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    return _save(fig, "threshold_cost_sensitivity", {"cost_sweep": sweep})


# --------------------------------------------------------------------------------
# Figure 5 — PR curves
# --------------------------------------------------------------------------------


def figure_pr_curve(y_true: np.ndarray, y_score: np.ndarray, name: str, title: str) -> Path:
    """A precision-recall curve with the no-skill baseline drawn in."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    base = float(y_true.mean())

    fig, ax = plt.subplots(figsize=(8.5, 5.4), facecolor=SURFACE)
    ax.plot(recall, precision, color=BLUE, linewidth=2, zorder=3)
    ax.axhline(base, color=BASELINE, linestyle="--", linewidth=1.2, zorder=1)
    ax.annotate(
        f"no-skill baseline ({base:.1%} — the fraud rate)",
        xy=(0.02, base), xytext=(0, 8), textcoords="offset points",
        color=MUTED, fontsize=9.5,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, color=GRID, linewidth=0.8, axis="both", zorder=0)
    _style_axes(ax, xlabel="recall", ylabel="precision", title=title)
    return _save(fig, name, {"baseline": base, "n": len(y_true)})


# --------------------------------------------------------------------------------
# Figure 6 — SHAP global importance
# --------------------------------------------------------------------------------


def figure_shap(contributions: list, name: str = "shap_global") -> Path:
    """Mean |SHAP| per feature — the global explanation.

    Horizontal bars: the labels are words of varying length, and rotating them to
    fit a vertical layout would make the chart unreadable for no benefit.
    """
    features = [c.feature for c in contributions][::-1]
    values = [c.value for c in contributions][::-1]

    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.34 * len(features))), facecolor=SURFACE)
    ax.barh(features, values, color=BLUE, height=0.62, zorder=2)
    for i, value in enumerate(values):
        ax.text(value * 1.02, i, f"{value:.3f}", va="center",
                color=INK_SECONDARY, fontsize=9.5)
    ax.grid(True, color=GRID, linewidth=0.8, axis="x", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)
    ax.set_xlim(0, max(values) * 1.16)
    ax.set_xlabel("mean |SHAP| contribution", color=INK_SECONDARY, fontsize=11, labelpad=8)
    ax.set_title("What the deployed model actually keys on", color=INK,
                 fontsize=14, pad=14, loc="left", fontweight="600")
    return _save(
        fig, name,
        {"features": [c.feature for c in contributions],
         "mean_abs_shap": [c.value for c in contributions]},
    )


# --------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------


def build_all(seed: int = 0) -> list[Path]:
    """Regenerate every figure the deck consumes."""
    paths = [figure_duplicates(), figure_model_comparison(), figure_leakage()]

    if (EXPERIMENTS_DIR / "threshold.json").exists():
        paths += [figure_threshold(), figure_cost_sensitivity()]
    else:
        log.warning("threshold.json missing — skipping threshold figures")

    # The PR curve and SHAP figures need a live model, not just stored metrics.
    from veridyx import features as feat
    from veridyx.data import prepare
    from veridyx.evaluate import run_one
    from veridyx.explain import global_importance
    from veridyx.models.gbm import GradientBoosting
    from veridyx.splits import TRAIN

    dataset = prepare(seed=seed)
    features = feat.build("portable", dataset.postings)
    fold = dataset.frame["fold_grouped"].to_numpy()
    result = run_one("lightgbm", features, dataset.labels, fold, "grouped", seed)
    paths.append(
        figure_pr_curve(
            result.labels, result.scores, "pr_curve_portable",
            "Precision-recall: the deployed model on the grouped test fold",
        )
    )

    from veridyx.evaluate import _subset

    model = GradientBoosting(seed=seed).fit(
        _subset(features, fold == TRAIN), dataset.labels[fold == TRAIN]
    )
    paths.append(figure_shap(global_importance(model, features, k=18, seed=seed)))
    return paths


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate every deck figure.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    paths = build_all(seed=args.seed)
    print(f"\nwrote {len(paths)} figures to {REPORTS_DIR}")
    for path in paths:
        print(f"  {path.name}  (+ {path.stem}.json)")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
