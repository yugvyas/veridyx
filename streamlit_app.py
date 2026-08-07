"""The hosted endpoint — composition only. The visual world lives in serve/theme.py.

At the repository root because Streamlit Community Cloud looks for the app there.
Scoring lives in serve/scoring.py and is framework-agnostic; this file decides what
appears where, and nothing about how the model behaves.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from serve import theme
from serve.scoring import (
    EXAMPLE_LEGIT,
    EXAMPLE_SCAM,
    get_deployment,
    score,
)

st.set_page_config(
    page_title="Veridyx — fraudulent job posting detection",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_, MANIFEST = get_deployment()
THRESHOLD = MANIFEST["operating_threshold"]

st.html(theme.css())

# Session state holds the form values so the worked examples can fill them, and holds
# the verdict so a rerun (a checkbox toggle, a resize) does not wipe the result the
# visitor is reading.
_DEFAULTS = {
    "title": "",
    "description": "",
    "location": "",
    "is_remote": False,
    "has_salary": False,
    "verdict": None,
    "error": None,
}
for key, value in _DEFAULTS.items():
    st.session_state.setdefault(key, value)


def _load_example(title: str, body: str, remote: bool, salary: bool, location: str = "") -> None:
    st.session_state.update(
        title=title,
        description=body,
        location=location,
        is_remote=remote,
        has_salary=salary,
        verdict=None,
        error=None,
    )


def _submit() -> None:
    try:
        st.session_state.verdict = score(
            st.session_state.title,
            st.session_state.description,
            st.session_state.location,
            st.session_state.is_remote,
            st.session_state.has_salary,
        )
        st.session_state.error = None
    except ValueError as exc:
        st.session_state.verdict = None
        st.session_state.error = str(exc)


st.html(theme.masthead(MANIFEST))

bench, tag = st.columns([1.06, 1], gap="large")

with bench:
    st.html('<div class="vx-fieldhead">Item under examination</div>')

    example_scam, example_legit = st.columns(2)
    example_scam.button(
        "Load a suspect posting",
        use_container_width=True,
        on_click=_load_example,
        args=("Work From Home Data Entry - Urgent", EXAMPLE_SCAM, True, False),
    )
    example_legit.button(
        "Load a genuine posting",
        use_container_width=True,
        on_click=_load_example,
        args=("Data Engineer, Platform", EXAMPLE_LEGIT, False, True, "Bangalore, India"),
    )

    st.text_input("Job title", key="title", placeholder="Data Analyst")
    # 230px pushed "Take a reading" below the fold on a 1440x900 laptop, which for a
    # Persuade surface means the primary action is invisible on arrival. 165px keeps
    # the whole control set inside the first viewport and still shows several lines.
    st.text_area("Description", key="description", height=165,
                 placeholder="Paste the body of the posting.")

    where, remote, salary = st.columns([2, 1, 1])
    where.text_input("Location", key="location", placeholder="Bangalore, India")
    remote.checkbox("Remote", key="is_remote")
    salary.checkbox("Pay stated", key="has_salary")

    with st.container(key="vx_submit"):
        st.button("Take a reading", type="primary",
                  use_container_width=True, on_click=_submit)

    if st.session_state.error:
        # The first version threw away str(exc) and printed the missing-title copy for
        # any ValueError, so a different failure would have been described wrongly. The
        # scorer's own message names the problem; this adds the recovery.
        # The recovery clause is conditional. Appending the title explanation to every
        # ValueError fixed the wrong *problem* and kept the wrong *recovery* — the same
        # defect one clause to the right.
        message = html.escape(st.session_state.error)
        if "title" in st.session_state.error.lower():
            message += (
                " — the model reads the title as well as the body, so scoring without"
                " one would not be the same measurement."
            )
        st.html(f'<p class="vx-error" role="alert">{message}</p>')

with tag:
    st.html('<div class="vx-fieldhead">Findings</div>')
    # aria-live so a screen reader is told a verdict arrived; the tag replaces itself
    # in place and nothing else signals the change.
    if st.session_state.verdict is not None:
        st.html(f'<div aria-live="polite">{theme.verdict_tag(st.session_state.verdict)}</div>')
    else:
        st.html(f'<div aria-live="polite">{theme.blank_tag(THRESHOLD, MANIFEST["model_version"])}</div>')

st.html('<div style="height:2.4rem"></div>')
st.html(theme.record(MANIFEST))
st.html(theme.colophon())
