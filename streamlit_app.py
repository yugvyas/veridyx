"""The hosted inference endpoint, on Streamlit Community Cloud.

At the repository root because Streamlit Community Cloud defaults to looking for the
app there; the scoring logic lives in `serve/scoring.py` and is framework-agnostic.

Originally built for Hugging Face Spaces. HF moved Gradio Spaces behind a paid tier
partway through, and because scoring was already separated from presentation, the
move cost a UI rewrite and nothing else.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from serve.scoring import EXAMPLE_LEGIT, EXAMPLE_SCAM, get_deployment, score

st.set_page_config(page_title="Veridyx", page_icon="🛡️", layout="wide")

_, MANIFEST = get_deployment()
THRESHOLD = MANIFEST["operating_threshold"]
METRICS = MANIFEST["test_metrics_at_operating_threshold"]

st.title("Veridyx")
st.caption("Fraudulent job posting detection — with the reasons behind every score.")

st.markdown(
    f"""
**Model** `{MANIFEST['model_version']}` · trained on {MANIFEST['training_rows']:,} EMSCAD
postings with near-duplicate clusters kept inside a single fold.
**Operating threshold** {THRESHOLD:.4f} — {MANIFEST['threshold_rationale']}.

At that threshold on the held-out test fold: precision **{METRICS['precision']:.3f}**,
recall **{METRICS['recall']:.3f}** — {METRICS['true_positives']} of {METRICS['n_positive']}
caught with **{METRICS['false_positives']} false alarms**.

The threshold is tuned for a review desk, so it is deliberately conservative: it flags
little and is right when it does. Recall is the price, and a different review capacity
implies a different threshold. That trade-off is the design, not a limitation.
"""
)

st.warning(
    "Trained on a 2012-2014 US-centric corpus. Against a live Indian job feed the score "
    "distribution shifts significantly (PSI 0.41), so scores on current non-US postings "
    "are not comparable to the metrics above. This is a screening aid that routes "
    "postings to a human reviewer — not a verdict.",
    icon="⚠️",
)

if "title" not in st.session_state:
    st.session_state.update(
        title="", description="", location="", is_remote=False, has_salary=False
    )


def _load_example(title: str, body: str, remote: bool, salary: bool, location: str = "") -> None:
    st.session_state.update(
        title=title, description=body, location=location,
        is_remote=remote, has_salary=salary,
    )


left, right = st.columns([3, 2], gap="large")

with left:
    example_a, example_b = st.columns(2)
    example_a.button(
        "Load a scam example", use_container_width=True,
        on_click=_load_example,
        args=("Work From Home Data Entry - Urgent", EXAMPLE_SCAM, True, False),
    )
    example_b.button(
        "Load a legitimate example", use_container_width=True,
        on_click=_load_example,
        args=("Data Engineer, Platform", EXAMPLE_LEGIT, False, True, "Bangalore, India"),
    )

    st.text_input("Job title", key="title", placeholder="Data Analyst")
    st.text_area("Description", key="description", height=260)
    field_a, field_b, field_c = st.columns([2, 1, 1])
    field_a.text_input("Location", key="location", placeholder="Bangalore, India")
    field_b.checkbox("Remote", key="is_remote")
    field_c.checkbox("Salary stated", key="has_salary")
    submitted = st.button("Score posting", type="primary", use_container_width=True)

with right:
    if not submitted:
        st.info("Enter a posting, or load one of the examples, then score it.")
    elif not st.session_state.title.strip():
        st.error("A job title is required.")
    else:
        verdict = score(
            st.session_state.title,
            st.session_state.description,
            st.session_state.location,
            st.session_state.is_remote,
            st.session_state.has_salary,
        )

        if verdict.flagged:
            st.error(f"### ⚑ Flag for review\nScore **{verdict.score:.4f}**", icon="🚩")
        else:
            st.success(f"### ○ Below review threshold\nScore **{verdict.score:.4f}**", icon="✅")
        st.progress(min(verdict.score, 1.0))
        st.caption(f"Operating threshold {verdict.threshold_used:.4f} · {verdict.model_version}")

        st.markdown("#### What drove this")
        st.dataframe(
            [
                {
                    "feature": c.feature,
                    "contribution": round(c.value, 3),
                    "direction": c.direction(),
                }
                for c in verdict.top_contributions
            ],
            hide_index=True,
            use_container_width=True,
        )

        with st.expander("Verdict (API response)"):
            st.json(verdict.model_dump(mode="json"))

st.divider()
st.caption(
    "Source and full methodology: github.com/yugvyas/veridyx · "
    "Sibling project: github.com/yugvyas/quantyx"
)
