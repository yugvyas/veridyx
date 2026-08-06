"""Train and persist the model that gets deployed, with its operating threshold.

The endpoint must not train on startup. A Space that fits a model on boot takes
minutes to become healthy, retrains on every restart, and — worst — can silently
serve a *different* model than the one the results table describes if anything about
the data or the code drifts underneath it.

So this module is the single place a deployable artifact is produced. It writes:

* `artifacts/lightgbm-portable.pkl` — the fitted model
* `artifacts/deployment.json` — the operating threshold, the metrics it was chosen
  against, and the model version, so the endpoint serves a decision it can justify

The threshold is chosen on the validation fold under the review-capacity framing,
never on test. Test is reported and then left alone.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from veridyx import features as feat
from veridyx.data import prepare
from veridyx.evaluate import _subset, metrics_at
from veridyx.models.base import ARTIFACT_DIR
from veridyx.models.gbm import GradientBoosting
from veridyx.splits import TEST, TRAIN, VAL
from veridyx.threshold import capacity_threshold, minimum_cost_threshold

log = logging.getLogger(__name__)

DEPLOYMENT_FILE = ARTIFACT_DIR / "deployment.json"


def train_deployable(seed: int = 0, capacity: int = 50) -> dict:
    """Fit the PORTABLE GBM, choose its threshold, persist both. Returns the manifest."""
    dataset = prepare(seed=seed)
    features = feat.build(feat.PORTABLE, dataset.postings)
    fold = dataset.frame["fold_grouped"].to_numpy()
    labels = dataset.labels

    model = GradientBoosting(seed=seed).fit(
        _subset(features, fold == TRAIN), labels[fold == TRAIN]
    )

    val_scores = model.predict_proba(_subset(features, fold == VAL))
    operating = capacity_threshold(labels[fold == VAL], val_scores, capacity)
    cost_choice = minimum_cost_threshold(labels[fold == VAL], val_scores)

    test_scores = model.predict_proba(_subset(features, fold == TEST))
    test_metrics = metrics_at(labels[fold == TEST], test_scores, operating.threshold)

    path = model.save(ARTIFACT_DIR / "lightgbm-portable.pkl")

    manifest = {
        "model_version": model.version,
        "artifact": path.name,
        "regime": feat.PORTABLE,
        "split": "grouped",
        "seed": seed,
        "operating_threshold": operating.threshold,
        "threshold_rationale": operating.rationale,
        "review_capacity": capacity,
        "alternative_threshold_minimum_cost": {
            "threshold": cost_choice.threshold,
            "rationale": cost_choice.rationale,
        },
        # Reported at the *deployed* threshold, not at the F1-optimal one, so the
        # numbers describe what the endpoint will actually do.
        "test_metrics_at_operating_threshold": {
            k: test_metrics[k]
            for k in ("pr_auc", "precision", "recall", "f1", "true_positives",
                      "false_positives", "false_negatives", "n", "n_positive")
        },
        "training_rows": int((fold == TRAIN).sum()),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    DEPLOYMENT_FILE.write_text(json.dumps(manifest, indent=2) + "\n")
    log.info("wrote %s and %s", path, DEPLOYMENT_FILE)
    return manifest


def load_deployment(artifact_dir: Path | None = None) -> tuple:
    """Load the persisted model and its manifest. Used by the endpoint."""
    from veridyx.models.base import load_model

    directory = artifact_dir or ARTIFACT_DIR
    manifest_path = directory / "deployment.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} missing. Build it with:\n"
            "  .venv/bin/python -m veridyx.train"
        )
    manifest = json.loads(manifest_path.read_text())
    model = load_model(directory / manifest["artifact"])
    return model, manifest


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and persist the deployable model.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--capacity", type=int, default=50)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    manifest = train_deployable(seed=args.seed, capacity=args.capacity)
    print(f"\ndeployable model: {manifest['model_version']}")
    print(f"  threshold {manifest['operating_threshold']:.4f} — {manifest['threshold_rationale']}")
    metrics = manifest["test_metrics_at_operating_threshold"]
    print(
        f"  at that threshold on test: P={metrics['precision']:.3f} "
        f"R={metrics['recall']:.3f} F1={metrics['f1']:.3f} "
        f"(caught {metrics['true_positives']} of {metrics['n_positive']}, "
        f"{metrics['false_positives']} false alarms)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
