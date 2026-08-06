"""The two feature regimes, and the boundary between them.

PORTABLE is computed from a `ScoreRequest` — the contract a live feed can satisfy.
FULL is computed from a `RawPosting` and is, structurally, *PORTABLE plus a block of
benchmark-only columns*: `full_features` calls `portable_features` and concatenates.

That composition is deliberate. It makes "FULL = PORTABLE + fields production does not
have" a property of the code rather than a claim in a docstring, so the two regimes
cannot silently drift apart and the reported gap between them stays meaningful.

The benchmark-only block includes `has_company_profile`, which is EMSCAD's single
strongest fraud signal and its most misleading one: a large share of fraudulent rows
simply left the field blank. No ATS feed exposes it. A model that leans on it scores
beautifully on the benchmark and cannot be deployed anywhere, which is the exact
failure this repository is built to demonstrate rather than commit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from veridyx.schema import RawPosting, ScoreRequest

# --------------------------------------------------------------------------------
# Hand-crafted signals, all computable from a live feed
# --------------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://|www\.")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE_RE = re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")
_MONEY_RE = re.compile(r"[$£€₹]|\b(?:usd|eur|gbp|inr)\b", re.IGNORECASE)
_ALLCAPS_RE = re.compile(r"\b[A-Z]{3,}\b")

# Phrases that recur in employment-scam copy. Kept small and legible on purpose: a
# long generated list would overfit EMSCAD's particular scams and is impossible to
# defend in a slide. These are here as interpretable features, not as a classifier —
# the models decide how much to trust them.
_SCAM_PHRASES = (
    "no experience",
    "work from home",
    "earn up to",
    "quick money",
    "start immediately",
    "wire transfer",
    "western union",
    "money transfer",
    "background check fee",
    "registration fee",
    "training fee",
    "unlimited earning",
    "be your own boss",
    "urgent hiring",
    "data entry",
    "personal assistant",
)

PORTABLE_FEATURE_NAMES: tuple[str, ...] = (
    "is_remote",
    "has_salary",
    "has_location",
    "text_length",
    "word_count",
    "title_length",
    "title_word_count",
    "uppercase_ratio",
    "title_uppercase_ratio",
    "allcaps_word_count",
    "exclamation_count",
    "digit_ratio",
    "has_url",
    "has_email",
    "has_phone",
    "has_money_symbol",
    "scam_phrase_count",
    "avg_word_length",
    "punctuation_ratio",
)

BENCHMARK_ONLY_FEATURE_NAMES: tuple[str, ...] = (
    "has_company_profile",
    "company_profile_length",
    "has_benefits",
    "has_requirements",
    "has_department",
    "has_company_logo",
    "has_questions",
    "has_employment_type",
    "has_required_experience",
    "has_required_education",
    "has_industry",
    "has_function",
)

FULL_FEATURE_NAMES: tuple[str, ...] = PORTABLE_FEATURE_NAMES + BENCHMARK_ONLY_FEATURE_NAMES


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _portable_row(request: ScoreRequest) -> list[float]:
    text = request.text()
    body = request.description or ""
    title = request.title
    lower = text.lower()

    alpha = sum(c.isalpha() for c in text)
    words = text.split()

    return [
        float(request.is_remote),
        float(request.has_salary),
        float(request.location is not None),
        float(len(text)),
        float(len(words)),
        float(len(title)),
        float(len(title.split())),
        _safe_ratio(sum(c.isupper() for c in text), alpha),
        _safe_ratio(sum(c.isupper() for c in title), sum(c.isalpha() for c in title)),
        float(len(_ALLCAPS_RE.findall(text))),
        float(text.count("!")),
        _safe_ratio(sum(c.isdigit() for c in text), len(text)),
        float(bool(_URL_RE.search(body))),
        float(bool(_EMAIL_RE.search(body))),
        float(bool(_PHONE_RE.search(body))),
        float(bool(_MONEY_RE.search(text))),
        float(sum(phrase in lower for phrase in _SCAM_PHRASES)),
        _safe_ratio(sum(len(w) for w in words), len(words)),
        _safe_ratio(sum(not c.isalnum() and not c.isspace() for c in text), len(text)),
    ]


def _benchmark_only_row(posting: RawPosting) -> list[float]:
    return [
        float(posting.company_profile is not None),
        float(len(posting.company_profile or "")),
        float(posting.benefits is not None),
        float(posting.requirements is not None),
        float(posting.department is not None),
        float(posting.has_company_logo),
        float(posting.has_questions),
        float(posting.employment_type is not None),
        float(posting.required_experience is not None),
        float(posting.required_education is not None),
        float(posting.industry is not None),
        float(posting.function is not None),
    ]


# --------------------------------------------------------------------------------
# Regimes
# --------------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureSet:
    """A dense numeric matrix plus the text corpus, with names attached.

    Text and numerics travel together because every model in Veridyx consumes both:
    the baseline vectorises `texts` and ignores `matrix`, the GBM uses both, and the
    transformer uses only `texts`. Keeping them in one object means a regime cannot
    be half-applied.
    """

    matrix: np.ndarray  # shape (n, len(names))
    names: tuple[str, ...]
    texts: list[str]
    regime: str

    def __post_init__(self) -> None:
        if self.matrix.shape != (len(self.texts), len(self.names)):
            raise ValueError(
                f"matrix shape {self.matrix.shape} does not match "
                f"{len(self.texts)} rows x {len(self.names)} names"
            )


PORTABLE = "portable"
FULL = "full"
REGIMES = (PORTABLE, FULL)


def portable_features(requests: list[ScoreRequest]) -> FeatureSet:
    """Features available anywhere a job posting can be read.

    Takes `ScoreRequest`, not `RawPosting`. The type signature is the enforcement:
    a benchmark-only column cannot reach this function to be accidentally used.
    """
    matrix = np.array([_portable_row(r) for r in requests], dtype=np.float64)
    if matrix.size == 0:
        matrix = matrix.reshape(0, len(PORTABLE_FEATURE_NAMES))
    return FeatureSet(
        matrix=matrix,
        names=PORTABLE_FEATURE_NAMES,
        texts=[r.text() for r in requests],
        regime=PORTABLE,
    )


def full_features(postings: list[RawPosting]) -> FeatureSet:
    """Everything EMSCAD offers. Matches the literature; deployable nowhere.

    Note the text corpus differs from PORTABLE's: here `company_profile` and
    `benefits` are concatenated into the document, which is what published TF-IDF and
    BERT results on this dataset are actually trained on.
    """
    requests = [p.to_score_request() for p in postings]
    base = portable_features(requests)
    extra = np.array([_benchmark_only_row(p) for p in postings], dtype=np.float64)
    if extra.size == 0:
        extra = extra.reshape(0, len(BENCHMARK_ONLY_FEATURE_NAMES))

    texts = [
        "\n\n".join(
            part
            for part in (req.text(), posting.company_profile, posting.benefits)
            if part
        )
        for req, posting in zip(requests, postings, strict=True)
    ]

    return FeatureSet(
        matrix=np.hstack([base.matrix, extra]),
        names=FULL_FEATURE_NAMES,
        texts=texts,
        regime=FULL,
    )


def build(regime: str, postings: list[RawPosting]) -> FeatureSet:
    """Dispatch by regime name. Used by the experiment runner."""
    if regime == PORTABLE:
        return portable_features([p.to_score_request() for p in postings])
    if regime == FULL:
        return full_features(postings)
    raise ValueError(f"unknown regime {regime!r}; expected one of {REGIMES}")
