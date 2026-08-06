"""Scoring for the hosted endpoint, independent of any UI framework.

Kept separate from the app so the thing being served is testable without spinning up
a web server, and so a change of hosting provider touches presentation only. That
separation earned itself immediately: Hugging Face moved Gradio Spaces behind a paid
tier mid-build, and the move to Streamlit changed no logic in this file.

Serves the PORTABLE LightGBM model. Not a fallback from DistilBERT — the only model in
the comparison that can be both served and explained. The transformer has no
attribution path, and the FULL model needs EMSCAD columns no caller can supply.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from veridyx.explain import explain_one
from veridyx.features import portable_features
from veridyx.schema import ScoreRequest, Verdict
from veridyx.train import load_deployment

log = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
# On a hosted deploy the repo is checked out whole, so artifacts/ sits at the root;
# a Space-style layout puts it beside the app. Both are tried.
ARTIFACT_CANDIDATES = (_HERE.parent / "artifacts", _HERE / "artifacts")

EXAMPLE_SCAM = """URGENT HIRING - Work from home data entry. No experience required!
Earn up to $5000 per week processing simple forms from your own computer.
Send your resume and a copy of your identification to begin immediately.
Payment made weekly by wire transfer. Registration fee applies. Limited positions!"""

EXAMPLE_LEGIT = """We are looking for a data engineer to join our platform team.
You will build and maintain batch and streaming pipelines, work closely with
analysts, and help evolve our warehouse model. Requires strong SQL, Python, and
experience with cloud data infrastructure at scale. Hybrid working, Bangalore."""


@lru_cache(maxsize=1)
def get_deployment() -> tuple:
    """Load the model and manifest once. Never trains — loads a committed artifact.

    An endpoint that fits a model on startup is slow to become healthy, refits on every
    restart, and can silently serve something other than what the results table
    describes.
    """
    for candidate in ARTIFACT_CANDIDATES:
        if (candidate / "deployment.json").exists():
            model, manifest = load_deployment(candidate)
            log.info("loaded %s from %s", manifest["model_version"], candidate)
            return model, manifest
    raise FileNotFoundError(
        "No deployment artifact found in "
        f"{[str(c) for c in ARTIFACT_CANDIDATES]}. Build one with:\n"
        "  .venv/bin/python -m veridyx.train"
    )


def score(
    title: str,
    description: str | None = None,
    location: str | None = None,
    is_remote: bool = False,
    has_salary: bool = False,
) -> Verdict:
    """Score one posting. Raises ValueError on an empty title."""
    if not (title or "").strip():
        raise ValueError("title is required")

    model, manifest = get_deployment()
    threshold = manifest["operating_threshold"]

    request = ScoreRequest(
        title=title,
        description=description or None,
        location=location or None,
        is_remote=bool(is_remote),
        has_salary=bool(has_salary),
    )
    features = portable_features([request])
    probability = float(model.predict_proba(features)[0])

    return Verdict(
        score=probability,
        flagged=probability >= threshold,
        threshold_used=threshold,
        model_version=model.version,
        top_contributions=explain_one(model, features, 0, k=8),
    )
