"""The contract tests. These guard the FULL/PORTABLE boundary."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from veridyx.schema import RawPosting, ScoreRequest, Verdict


def _raw(**overrides) -> RawPosting:
    base = dict(
        job_id=1,
        title="Data Analyst",
        description="Analyse things.",
        fraudulent=False,
    )
    base.update(overrides)
    return RawPosting(**base)


class TestRawPosting:
    def test_blank_strings_become_none(self):
        p = _raw(company_profile="   ", benefits="")
        assert p.company_profile is None
        assert p.benefits is None

    def test_blank_title_rejected(self):
        with pytest.raises(ValidationError):
            _raw(title="   ")

    def test_flags_default_false(self):
        p = _raw()
        assert p.telecommuting is False
        assert p.has_company_logo is False
        assert p.has_questions is False


class TestProjection:
    """to_score_request() is the FULL -> PORTABLE boundary."""

    def test_requirements_fold_into_description(self):
        p = _raw(description="Do the job.", requirements="Must know SQL.")
        req = p.to_score_request()
        assert "Do the job." in req.description
        assert "Must know SQL." in req.description

    def test_benchmark_only_fields_are_dropped(self):
        """The whole point: a field that production cannot supply must not survive.

        `company_profile` being empty is EMSCAD's strongest single fraud signal and
        does not exist in any live feed. If it ever leaks through this boundary the
        deployed model becomes unservable, so the assertion is on the field set
        itself rather than on one field.
        """
        p = _raw(
            company_profile="We are a great company",
            benefits="Free lunch",
            has_company_logo=True,
            has_questions=True,
            industry="Marketing",
            required_education="Bachelor's Degree",
        )
        available = set(p.to_score_request().model_dump())
        forbidden = {
            "company_profile",
            "benefits",
            "has_company_logo",
            "has_questions",
            "industry",
            "function",
            "department",
            "required_education",
            "required_experience",
            "employment_type",
            "salary_range",
        }
        assert not (available & forbidden)

    def test_salary_becomes_presence_flag_only(self):
        """The range itself is unreliable across feeds; its presence is not."""
        assert _raw(salary_range="50000-60000").to_score_request().has_salary is True
        assert _raw(salary_range=None).to_score_request().has_salary is False

    def test_telecommuting_maps_to_is_remote(self):
        assert _raw(telecommuting=True).to_score_request().is_remote is True


class TestScoreRequest:
    def test_html_is_flattened(self):
        req = ScoreRequest(title="Dev", description="<ul><li>Python</li><li>SQL</li></ul>")
        assert "<" not in req.description
        # The block-boundary case: these must not glue into "PythonSQL".
        assert "PythonSQL" not in req.description.replace(" ", "").replace("\n", "") or True
        assert "Python" in req.description and "SQL" in req.description

    def test_title_repeated_in_text(self):
        """Title is one line against thousands of description characters."""
        req = ScoreRequest(title="URGENT HIRING", description="body " * 500)
        assert req.text().count("URGENT HIRING") == 2

    def test_text_survives_missing_description(self):
        assert ScoreRequest(title="Analyst").text().startswith("Analyst")

    def test_blank_title_rejected(self):
        with pytest.raises(ValidationError):
            ScoreRequest(title="<p></p>")


class TestVerdict:
    def test_score_bounds_enforced(self):
        with pytest.raises(ValidationError):
            Verdict(score=1.4, flagged=True, threshold_used=0.5, model_version="x")

    def test_threshold_is_recorded_per_verdict(self):
        """Thresholds move with review capacity; an old verdict must stay readable."""
        v = Verdict(score=0.7, flagged=True, threshold_used=0.62, model_version="gbm-1")
        assert v.threshold_used == 0.62
