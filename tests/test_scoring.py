"""Endpoint scoring tests.

These exercise the deployed artifact rather than a freshly-trained model, because the
thing that can silently break in production is the artifact — a stale pickle, a
manifest whose threshold no longer matches, a model saved under the wrong regime. A
test that retrains would pass while all three were broken.

Skipped when no artifact has been built, so a clean clone can still run the suite.
"""

from __future__ import annotations

import pytest

from veridyx.features import PORTABLE
from veridyx.schema import Verdict

scoring = pytest.importorskip("serve.scoring")

pytestmark = pytest.mark.skipif(
    not any((c / "deployment.json").exists() for c in scoring.ARTIFACT_CANDIDATES),
    reason="no deployment artifact; build one with `python -m veridyx.train`",
)


@pytest.fixture(scope="module")
def deployment():
    return scoring.get_deployment()


class TestDeploymentArtifact:
    def test_serves_the_portable_regime(self, deployment):
        """A FULL model here would be unservable: callers cannot supply its columns."""
        model, manifest = deployment
        assert model.regime == PORTABLE
        assert manifest["regime"] == PORTABLE

    def test_manifest_threshold_is_a_probability(self, deployment):
        _, manifest = deployment
        assert 0.0 <= manifest["operating_threshold"] <= 1.0

    def test_manifest_metrics_match_the_deployed_threshold(self, deployment):
        """The published numbers must describe what the endpoint actually does.

        Reporting F1-optimal metrics beside a capacity-chosen threshold would be a
        quiet lie: the endpoint would behave nothing like its own documentation.
        """
        _, manifest = deployment
        metrics = manifest["test_metrics_at_operating_threshold"]
        assert metrics["true_positives"] + metrics["false_negatives"] == metrics["n_positive"]
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0

    def test_loading_is_cached(self, deployment):
        """The endpoint must not reload a 4 MB pickle per request."""
        assert scoring.get_deployment() is deployment


class TestScoring:
    def test_scam_example_is_flagged(self):
        verdict = scoring.score(
            "Work From Home Data Entry - Urgent", scoring.EXAMPLE_SCAM, is_remote=True
        )
        assert isinstance(verdict, Verdict)
        assert verdict.flagged
        assert verdict.score > verdict.threshold_used

    def test_legitimate_example_is_not_flagged(self):
        verdict = scoring.score(
            "Data Engineer, Platform", scoring.EXAMPLE_LEGIT,
            location="Bangalore, India", has_salary=True,
        )
        assert not verdict.flagged
        assert verdict.score < verdict.threshold_used

    def test_the_two_examples_are_far_apart(self):
        """Guards against an artifact that loads but predicts a constant."""
        scam = scoring.score("Work From Home", scoring.EXAMPLE_SCAM).score
        legit = scoring.score("Data Engineer", scoring.EXAMPLE_LEGIT).score
        assert scam - legit > 0.5

    def test_every_verdict_carries_its_justification(self):
        """A flag a reviewer cannot interrogate is one they will rubber-stamp."""
        verdict = scoring.score("Work From Home Data Entry", scoring.EXAMPLE_SCAM)
        assert verdict.top_contributions
        assert verdict.model_version
        assert all(c.feature and c.value != 0 for c in verdict.top_contributions)

    def test_flag_decision_follows_the_threshold(self):
        verdict = scoring.score("Data Engineer", scoring.EXAMPLE_LEGIT)
        assert verdict.flagged == (verdict.score >= verdict.threshold_used)

    def test_empty_title_rejected(self):
        with pytest.raises(ValueError, match="title is required"):
            scoring.score("   ", "some description")

    def test_description_is_optional(self):
        assert isinstance(scoring.score("Data Analyst"), Verdict)

    def test_html_description_is_accepted(self):
        """ATS feeds send markup; the endpoint must not need it pre-cleaned."""
        verdict = scoring.score("Engineer", "<ul><li>Python</li><li>SQL</li></ul>")
        assert 0.0 <= verdict.score <= 1.0
