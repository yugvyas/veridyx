# serve/

The hosted inference endpoint.

- `scoring.py` — framework-agnostic scoring. Loads the committed artifact from
  `artifacts/`, returns a full `Verdict` (score, decision, SHAP contributions, model
  version). Never trains on startup.
- The UI lives at `streamlit_app.py` in the repository root, because Streamlit
  Community Cloud looks for it there.

## Why Streamlit rather than Hugging Face Spaces

This was built for HF Spaces first. Partway through, HF moved Gradio and Docker Spaces
behind a PRO subscription — only static Spaces remain free — so the endpoint moved to
Streamlit Community Cloud, which is free and deploys from this repository directly.

Because scoring was already separated from presentation, that change cost a UI rewrite
and touched no scoring logic. `serve/scoring.py` is byte-identical across the move.

## Running locally

    .venv/bin/streamlit run streamlit_app.py

## What is served

The PORTABLE LightGBM model — the only model in the comparison that can be both served
and explained. DistilBERT is the ceiling check and has no attribution path; the FULL
model reaches a higher benchmark score (PR-AUC 0.883 vs 0.773) using EMSCAD columns
that no live feed provides, so it is unservable by construction. That ~0.11 PR-AUC gap
is the measured cost of deployability.
