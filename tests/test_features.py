"""Feature regime tests.

The central assertion is structural: FULL must be exactly PORTABLE plus a
benchmark-only block, in that column order. If that ever stops holding, the reported
"regime gap" is no longer measuring what it claims to measure.
"""

from __future__ import annotations

import numpy as np
import pytest

from veridyx.features import (
    BENCHMARK_ONLY_FEATURE_NAMES,
    FULL,
    FULL_FEATURE_NAMES,
    PORTABLE,
    PORTABLE_FEATURE_NAMES,
    build,
    full_features,
    portable_features,
)
from veridyx.schema import RawPosting


def _raw(**overrides) -> RawPosting:
    base = dict(
        job_id=1,
        title="Data Analyst",
        description="Analyse things carefully and report findings to the team.",
        fraudulent=False,
    )
    base.update(overrides)
    return RawPosting(**base)


class TestRegimeComposition:
    def test_full_is_portable_plus_benchmark_only(self):
        assert FULL_FEATURE_NAMES == PORTABLE_FEATURE_NAMES + BENCHMARK_ONLY_FEATURE_NAMES

    def test_full_matrix_prefix_equals_portable_matrix(self):
        """Not just the names — the values must line up column for column."""
        postings = [_raw(job_id=i, company_profile="x" * i) for i in range(1, 6)]
        portable = portable_features([p.to_score_request() for p in postings])
        full = full_features(postings)
        n = len(PORTABLE_FEATURE_NAMES)
        assert np.allclose(full.matrix[:, :n], portable.matrix)

    def test_regimes_are_labelled(self):
        postings = [_raw()]
        assert portable_features([postings[0].to_score_request()]).regime == PORTABLE
        assert full_features(postings).regime == FULL

    def test_build_dispatches(self):
        postings = [_raw()]
        assert build(PORTABLE, postings).regime == PORTABLE
        assert build(FULL, postings).regime == FULL

    def test_build_rejects_unknown_regime(self):
        with pytest.raises(ValueError, match="unknown regime"):
            build("semi-portable", [_raw()])


class TestPortableSignals:
    def test_company_profile_cannot_influence_portable(self):
        """The leak test. EMSCAD's strongest signal must not reach PORTABLE."""
        without = portable_features([_raw(company_profile=None).to_score_request()])
        with_profile = portable_features(
            [_raw(company_profile="A long and reassuring company story.").to_score_request()]
        )
        assert np.allclose(without.matrix, with_profile.matrix)
        assert without.texts == with_profile.texts

    def test_scam_phrases_counted(self):
        idx = PORTABLE_FEATURE_NAMES.index("scam_phrase_count")
        scam = _raw(
            description="Work from home, no experience needed, earn up to $9000 by wire transfer."
        )
        legit = _raw(description="You will maintain our data warehouse and mentor juniors.")
        assert portable_features([scam.to_score_request()]).matrix[0, idx] > 0
        assert portable_features([legit.to_score_request()]).matrix[0, idx] == 0

    def test_contact_details_detected_in_body(self):
        names = PORTABLE_FEATURE_NAMES
        req = _raw(
            description="Email hr@example.com or call +1 555 123 4567, see https://example.com"
        ).to_score_request()
        row = portable_features([req]).matrix[0]
        assert row[names.index("has_email")] == 1.0
        assert row[names.index("has_phone")] == 1.0
        assert row[names.index("has_url")] == 1.0

    def test_uppercase_ratio_flags_shouting_titles(self):
        idx = PORTABLE_FEATURE_NAMES.index("title_uppercase_ratio")
        loud = portable_features([_raw(title="URGENT HIRING NOW").to_score_request()])
        calm = portable_features([_raw(title="Data Analyst").to_score_request()])
        assert loud.matrix[0, idx] > calm.matrix[0, idx]

    def test_no_nan_or_inf_on_degenerate_input(self):
        """Ratios divide by counts that can legitimately be zero."""
        req = _raw(title="A", description=None).to_score_request()
        matrix = portable_features([req]).matrix
        assert np.isfinite(matrix).all()


class TestFullSignals:
    def test_missing_company_profile_is_visible_to_full(self):
        idx = FULL_FEATURE_NAMES.index("has_company_profile")
        assert full_features([_raw(company_profile=None)]).matrix[0, idx] == 0.0
        assert full_features([_raw(company_profile="We are great")]).matrix[0, idx] == 1.0

    def test_full_text_includes_company_profile(self):
        """Published TF-IDF/BERT results on EMSCAD train on this concatenation."""
        text = full_features([_raw(company_profile="UNIQUEPROFILETOKEN")]).texts[0]
        assert "UNIQUEPROFILETOKEN" in text

    def test_portable_text_excludes_company_profile(self):
        req = _raw(company_profile="UNIQUEPROFILETOKEN").to_score_request()
        assert "UNIQUEPROFILETOKEN" not in portable_features([req]).texts[0]


class TestFeatureSetIntegrity:
    def test_shape_mismatch_is_rejected(self):
        from veridyx.features import FeatureSet

        with pytest.raises(ValueError, match="does not match"):
            FeatureSet(
                matrix=np.zeros((2, 3)),
                names=PORTABLE_FEATURE_NAMES,
                texts=["a", "b"],
                regime=PORTABLE,
            )

    def test_empty_input_keeps_column_count(self):
        assert portable_features([]).matrix.shape == (0, len(PORTABLE_FEATURE_NAMES))
        assert full_features([]).matrix.shape == (0, len(FULL_FEATURE_NAMES))
