"""The contract every component of Veridyx speaks.

Three types, and the boundary between the first two is the most important design
decision in this repository:

* `RawPosting` — one EMSCAD row. Carries the benchmark's full column set, including
  fields that exist *only* in the benchmark (`company_profile`, `has_company_logo`,
  `has_questions`, `benefits`, ...).

* `ScoreRequest` — what a model is actually asked to judge in production. It carries
  **only** fields a live ATS or aggregator feed provides. This is not a convenience
  subset; it is the reason the deployed model and the quantyx bridge and the review
  sheet are one code path instead of three.

* `Verdict` — a score plus everything needed to defend it: the threshold that
  produced the label, the features that drove it, and the model version.

The asymmetry is deliberate. A model trained on `RawPosting` cannot be deployed
anywhere. A model trained on `ScoreRequest.text()` can be deployed everywhere, and
the difference between their scores is a measurement worth reporting.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field, field_validator

from veridyx.text import html_to_text

# --------------------------------------------------------------------------------
# Benchmark side
# --------------------------------------------------------------------------------


class RawPosting(BaseModel):
    """One row of EMSCAD, normalized. Blank strings become None throughout.

    EMSCAD encodes missingness as the empty string in some columns and as a true
    null in others, depending on which export you pull. Collapsing both to None here
    means no downstream component has to know which — and, critically, means the
    `company_profile`-is-empty signal is measured consistently rather than depending
    on the export.
    """

    job_id: int
    title: str
    location: str | None = None
    department: str | None = None
    salary_range: str | None = None

    company_profile: str | None = None
    description: str | None = None
    requirements: str | None = None
    benefits: str | None = None

    telecommuting: bool = False
    has_company_logo: bool = False
    has_questions: bool = False

    employment_type: str | None = None
    required_experience: str | None = None
    required_education: str | None = None
    industry: str | None = None
    function: str | None = None

    fraudulent: bool

    @field_validator(
        "location",
        "department",
        "salary_range",
        "company_profile",
        "description",
        "requirements",
        "benefits",
        "employment_type",
        "required_experience",
        "required_education",
        "industry",
        "function",
        mode="before",
    )
    @classmethod
    def _blank_to_none(cls, v: object) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s or None

    @field_validator("title", mode="before")
    @classmethod
    def _require_title(cls, v: object) -> str:
        s = str(v).strip() if v is not None else ""
        if not s:
            raise ValueError("title must be a non-empty string")
        return s

    def to_score_request(self) -> ScoreRequest:
        """Project onto the deployable contract.

        `requirements` folds into the description because a live feed delivers one
        text blob — an ATS does not hand you requirements as a separate column. Not
        folding it would give the PORTABLE model information production never has.
        """
        body = "\n\n".join(p for p in (self.description, self.requirements) if p)
        return ScoreRequest(
            title=self.title,
            description=body or None,
            location=self.location,
            is_remote=self.telecommuting,
            has_salary=self.salary_range is not None,
            source_id=str(self.job_id),
        )


# --------------------------------------------------------------------------------
# Deployable side
# --------------------------------------------------------------------------------


class ScoreRequest(BaseModel):
    """A posting as production sees it.

    Every field here exists in EMSCAD *and* in quantyx's `Posting` *and* in a bare
    ATS payload. Adding a field to this model is a load-bearing decision: it must be
    obtainable from a live feed, or the deployed model becomes unservable.

    `company` is carried for display and audit only. It is deliberately not a
    feature — EMSCAD has no company-name column, so training on it is impossible
    and pretending otherwise would produce a model that cannot be evaluated.
    """

    title: str
    description: str | None = None
    location: str | None = None
    is_remote: bool = False
    has_salary: bool = False

    company: str | None = None
    url: str | None = None
    source_id: str | None = None

    @field_validator("description", mode="before")
    @classmethod
    def _flatten_html(cls, v: object) -> str | None:
        """Accept raw HTML. ATS feeds send markup; EMSCAD sends near-plain text."""
        if v is None:
            return None
        return html_to_text(str(v))

    @field_validator("title", mode="before")
    @classmethod
    def _clean_title(cls, v: object) -> str:
        s = html_to_text(str(v)) if v is not None else None
        if not s:
            raise ValueError("title must be a non-empty string")
        return s

    def text(self) -> str:
        """The single text field every model consumes.

        Title is repeated deliberately. It is one line against a description of
        several thousand characters, and a bare concatenation lets TF-IDF drown it —
        but the title is where "EARN $$$ FROM HOME" lives.
        """
        return f"{self.title}\n{self.title}\n\n{self.description or ''}".strip()


# --------------------------------------------------------------------------------
# Output side
# --------------------------------------------------------------------------------


class Contribution(BaseModel):
    """One feature's signed push on a single prediction, from SHAP."""

    feature: str
    value: float

    def direction(self) -> str:
        return "toward fraud" if self.value > 0 else "toward legitimate"


class Verdict(BaseModel):
    """A scored posting, carrying enough context to defend the call.

    `threshold_used` is stored per-verdict rather than looked up globally because the
    threshold is a *cost decision* that changes with review capacity. A verdict
    recorded last week under a different capacity must remain interpretable.
    """

    score: float = Field(ge=0.0, le=1.0)
    flagged: bool
    threshold_used: float = Field(ge=0.0, le=1.0)
    model_version: str
    top_contributions: list[Contribution] = Field(default_factory=list)

    source_id: str | None = None
    scored_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ReviewVerdict(BaseModel):
    """A human's ruling on a flagged posting. Appended to verdicts.jsonl.

    This is the artifact that turns "human in the loop" from a claim into evidence,
    so it records what the model said at the time — a later retrain must not be able
    to rewrite the history of what a reviewer was actually shown.
    """

    source_id: str
    decision: str  # "fraud" | "legitimate" | "unclear"
    model_score: float
    model_version: str
    note: str | None = None
    reviewed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("decision")
    @classmethod
    def _known_decision(cls, v: str) -> str:
        allowed = {"fraud", "legitimate", "unclear"}
        if v not in allowed:
            raise ValueError(f"decision must be one of {sorted(allowed)}")
        return v
