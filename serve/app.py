"""The hosted inference endpoint — Gradio UI plus a JSON API, on Hugging Face Spaces.

Serves the PORTABLE LightGBM model. That is not a fallback from DistilBERT; it is the
only model in the comparison that can be served *and* explained. The transformer is
the ceiling check and has no attribution path (see `veridyx.explain`), and the FULL
model depends on `company_profile` and friends, which no caller can supply.

Every response is a `Verdict`: a score, a decision against the deployed operating
threshold, the SHAP contributions that produced it, and the model version. A flag a
caller cannot interrogate is a flag they will rubber-stamp or ignore.

The model is loaded once at import from a committed artifact — never trained on boot.
A Space that fits on startup is slow to become healthy and can silently serve a model
that no longer matches the published results.

Run locally:   .venv/bin/python serve/app.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import gradio as gr

# On a Space, `veridyx` is pip-installed from requirements.txt and this is a no-op.
# Locally, `python serve/app.py` puts serve/ on sys.path rather than the repo root, so
# the documented run command would fail on import without this.
_HERE = Path(__file__).resolve().parent
if not (_HERE / "veridyx").exists():
    sys.path.insert(0, str(_HERE.parent))

from veridyx.explain import explain_one  # noqa: E402
from veridyx.features import portable_features  # noqa: E402
from veridyx.schema import ScoreRequest, Verdict  # noqa: E402
from veridyx.train import load_deployment  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# On a Space the artifacts sit beside app.py; locally they are in the repo root.
ARTIFACT_CANDIDATES = (_HERE / "artifacts", _HERE.parent / "artifacts")

MODEL, MANIFEST = None, None
for candidate in ARTIFACT_CANDIDATES:
    if (candidate / "deployment.json").exists():
        MODEL, MANIFEST = load_deployment(candidate)
        log.info("loaded %s from %s", MANIFEST["model_version"], candidate)
        break
if MODEL is None:
    raise FileNotFoundError(
        "No deployment artifact found. Build one with:\n"
        "  .venv/bin/python -m veridyx.train"
    )

THRESHOLD = MANIFEST["operating_threshold"]

EXAMPLE_SCAM = """URGENT HIRING - Work from home data entry. No experience required!
Earn up to $5000 per week processing simple forms from your own computer.
Send your resume and a copy of your identification to begin immediately.
Payment made weekly by wire transfer. Registration fee applies. Limited positions!"""

EXAMPLE_LEGIT = """We are looking for a data engineer to join our platform team.
You will build and maintain batch and streaming pipelines, work closely with
analysts, and help evolve our warehouse model. Requires strong SQL, Python, and
experience with cloud data infrastructure at scale. Hybrid working, Bangalore."""


def score(title: str, description: str, location: str, is_remote: bool, has_salary: bool) -> dict:
    """Score one posting. Returns a `Verdict` as a plain dict."""
    if not (title or "").strip():
        return {"error": "title is required"}

    request = ScoreRequest(
        title=title,
        description=description or None,
        location=location or None,
        is_remote=bool(is_remote),
        has_salary=bool(has_salary),
    )
    features = portable_features([request])
    probability = float(MODEL.predict_proba(features)[0])
    contributions = explain_one(MODEL, features, 0, k=8)

    return Verdict(
        score=probability,
        flagged=probability >= THRESHOLD,
        threshold_used=THRESHOLD,
        model_version=MODEL.version,
        top_contributions=contributions,
    ).model_dump(mode="json")


def _score_for_ui(title, description, location, is_remote, has_salary):
    verdict = score(title, description, location, is_remote, has_salary)
    if "error" in verdict:
        return {}, "", verdict

    label = {
        "flag for review": verdict["score"],
        "no action": 1 - verdict["score"],
    }
    headline = (
        f"### {'⚑ Flag for review' if verdict['flagged'] else '○ Below review threshold'}\n"
        f"Score **{verdict['score']:.4f}** against a threshold of **{THRESHOLD:.4f}**.\n\n"
        "**What drove this**\n\n"
        + "\n".join(
            f"- `{c['feature']}` {c['value']:+.2f} — {'toward fraud' if c['value'] > 0 else 'toward legitimate'}"
            for c in verdict["top_contributions"]
        )
    )
    return label, headline, verdict


_METRICS = MANIFEST["test_metrics_at_operating_threshold"]
_DESCRIPTION = f"""
Scores a job posting for signs of employment fraud, and shows the features behind the score.

**Model** `{MANIFEST['model_version']}` · trained on {MANIFEST['training_rows']:,} EMSCAD postings
with near-duplicate clusters kept inside a single fold.
**Operating threshold** {THRESHOLD:.4f} — {MANIFEST['threshold_rationale']}.
At that threshold on the held-out test fold: precision **{_METRICS['precision']:.3f}**,
recall **{_METRICS['recall']:.3f}** ({_METRICS['true_positives']} of {_METRICS['n_positive']} caught,
{_METRICS['false_positives']} false alarms).

The threshold is tuned for a review desk, so it is deliberately conservative: it flags
little and is right when it does. Recall is the price. A different review capacity
implies a different threshold — that trade-off is the point, not a limitation.

⚠︎ This model was trained on a 2012-2014 US-centric corpus. Scores on current postings
from other markets are not directly comparable; the repository's drift report measures
how far they have moved (PSI 0.41 against a live Indian job feed).
"""

with gr.Blocks(title="Veridyx — fraudulent job posting detection") as demo:
    gr.Markdown("# Veridyx")
    gr.Markdown(_DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=3):
            title_in = gr.Textbox(label="Job title", placeholder="Data Analyst")
            description_in = gr.Textbox(label="Description", lines=12)
            with gr.Row():
                location_in = gr.Textbox(label="Location", placeholder="Bangalore, India")
                remote_in = gr.Checkbox(label="Remote")
                salary_in = gr.Checkbox(label="Salary stated")
            submit = gr.Button("Score posting", variant="primary")
        with gr.Column(scale=2):
            label_out = gr.Label(label="Decision", num_top_classes=2)
            explain_out = gr.Markdown()
            json_out = gr.JSON(label="Verdict (API response)")

    gr.Examples(
        examples=[
            ["Work From Home Data Entry - Urgent", EXAMPLE_SCAM, "", True, False],
            ["Data Engineer, Platform", EXAMPLE_LEGIT, "Bangalore, India", False, True],
        ],
        inputs=[title_in, description_in, location_in, remote_in, salary_in],
    )

    submit.click(
        _score_for_ui,
        inputs=[title_in, description_in, location_in, remote_in, salary_in],
        outputs=[label_out, explain_out, json_out],
        api_name="score",
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
