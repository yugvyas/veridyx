"""Attribution tests.

These train real (tiny) models rather than mocking, because the failure this file
guards against is a misalignment between contribution columns and feature names —
exactly the kind of bug a mock reproduces perfectly while the real model does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from veridyx.explain import (
    ExplanationError,
    contribution_matrix,
    explain,
    global_importance,
)
from veridyx.features import PORTABLE, portable_features
from veridyx.models.baseline import TfidfLogisticRegression
from veridyx.models.bert import DistilBertClassifier
from veridyx.models.gbm import GradientBoosting
from veridyx.schema import ScoreRequest

# Shared boilerplate. Without it the two classes have disjoint vocabularies, the data
# becomes perfectly separable, and LightGBM correctly solves it with a *single* split —
# so every explanation contains exactly one feature and any test of attribution
# richness fails against a model that is behaving properly. Real postings share most
# of their language; the fixture has to as well.
_SHARED = (
    "Thank you for your interest in this position. We review every application and "
    "aim to respond to shortlisted candidates within two weeks. This role reports to "
    "a team lead and involves regular communication with colleagues across the "
    "organisation. Please read the description below in full before applying, and "
    "make sure your contact details are current so we can reach you."
)
_FRAUD_BODY = (
    "Earn cash fast from home with no experience required. Wire transfer payment "
    "weekly, urgent hiring, send your identification documents to start immediately. "
    "Unlimited earning potential for motivated applicants who can start right away."
)
_LEGIT_BODY = (
    "We are hiring a backend engineer to maintain our payment services. You will "
    "work with Python, PostgreSQL and Kubernetes alongside a small platform team. "
    "Experience with distributed systems and a pragmatic approach to testing matter."
)


@pytest.fixture(scope="module")
def corpus():
    """Overlapping classes, so a tree must use several features to separate them."""
    rng = np.random.default_rng(0)
    requests, labels = [], []
    for i in range(80):
        fraud = i % 2 == 0
        body = _FRAUD_BODY if fraud else _LEGIT_BODY
        # A tenth of each class borrows the other's boilerplate, which stops the split
        # from being trivially perfect and forces the model to weigh evidence.
        if rng.random() < 0.10:
            body = f"{body} {_LEGIT_BODY if fraud else _FRAUD_BODY}"
        requests.append(
            ScoreRequest(
                title="Work From Home Earner" if fraud else "Backend Engineer",
                description=f"{_SHARED} {body} reference number {i}",
            )
        )
        labels.append(fraud)
    return portable_features(requests), np.array(labels, dtype=bool)


@pytest.fixture(scope="module")
def gbm(corpus):
    features, labels = corpus
    return GradientBoosting(n_estimators=40, max_features=500, seed=0).fit(features, labels)


@pytest.fixture(scope="module")
def linear(corpus):
    features, labels = corpus
    return TfidfLogisticRegression(max_features=500, min_df=1, seed=0).fit(features, labels)


class TestContributionMatrix:
    @pytest.mark.parametrize("model_name", ["gbm", "linear"])
    def test_shape_matches_names(self, model_name, corpus, request):
        model = request.getfixturevalue(model_name)
        features, _ = corpus
        matrix, names = contribution_matrix(model, features)
        assert matrix.shape[0] == len(features.texts)
        assert matrix.shape[1] == len(names)

    def test_gbm_sparse_return_is_handled(self, gbm, corpus):
        """LightGBM returns sparse contributions for sparse input.

        `np.asarray` on that gives a 0-d object array, not a 2-D matrix — the shape
        looks right until it is indexed. A non-degenerate shape here is the guard.
        """
        features, _ = corpus
        matrix, _ = contribution_matrix(gbm, features)
        assert matrix.ndim == 2
        assert np.isfinite(matrix).all()
        assert np.abs(matrix).sum() > 0


class TestExplanations:
    @pytest.mark.parametrize("model_name", ["gbm", "linear"])
    def test_contributions_are_faithful_to_the_score(self, model_name, corpus, request):
        """The defining SHAP property: contributions sum to the prediction.

        An earlier version of this test asserted that the explanation contained
        specific scam words. That tests the *model*, not the explainer, and it failed
        against a perfectly correct GBM which had found one feature sufficient. What
        actually has to hold is local accuracy — the contributions must add up to what
        the model did, up to a constant base value. If the column-to-name mapping is
        misaligned, or a batch boundary drops rows, this correlation collapses.
        """
        model = request.getfixturevalue(model_name)
        features, _ = corpus
        matrix, _ = contribution_matrix(model, features)
        totals = matrix.sum(axis=1)
        scores = model.predict_proba(features)
        # Contributions live in log-odds space, probabilities do not, so the
        # relationship is monotone rather than linear; rank correlation is the honest
        # check. Anything below ~0.99 means the attribution is not tracking the model.
        order_totals = np.argsort(np.argsort(totals))
        order_scores = np.argsort(np.argsort(scores))
        correlation = np.corrcoef(order_totals, order_scores)[0, 1]
        assert correlation > 0.99, f"attribution does not track the model ({correlation:.3f})"

    @pytest.mark.parametrize("model_name", ["gbm", "linear"])
    def test_fraud_rows_attract_more_positive_attribution(self, model_name, corpus, request):
        """Aggregate direction, rather than any particular word."""
        model = request.getfixturevalue(model_name)
        features, labels = corpus
        matrix, _ = contribution_matrix(model, features)
        totals = matrix.sum(axis=1)
        assert totals[labels].mean() > totals[~labels].mean()

    def test_top_k_is_honoured_and_ordered(self, gbm, corpus):
        features, _ = corpus
        for contribs in explain(gbm, features, k=5):
            assert len(contribs) <= 5
            magnitudes = [abs(c.value) for c in contribs]
            assert magnitudes == sorted(magnitudes, reverse=True)

    def test_zero_contributions_are_dropped(self, gbm, corpus):
        """A feature that did nothing is noise on a review sheet."""
        features, _ = corpus
        assert all(c.value != 0.0 for contribs in explain(gbm, features) for c in contribs)

    def test_direction_reads_correctly(self, gbm, corpus):
        features, _ = corpus
        for contribs in explain(gbm, features, k=3):
            for c in contribs:
                expected = "toward fraud" if c.value > 0 else "toward legitimate"
                assert c.direction() == expected

    def test_tfidf_prefix_is_stripped_for_display(self, gbm, corpus):
        features, _ = corpus
        for contribs in explain(gbm, features, k=8):
            assert all(not c.feature.startswith("term:") for c in contribs)


class TestGlobalImportance:
    def test_returns_ranked_magnitudes(self, gbm, corpus):
        features, _ = corpus
        importance = global_importance(gbm, features, k=10)
        values = [c.value for c in importance]
        assert values == sorted(values, reverse=True)
        assert all(v >= 0 for v in values)

    def test_sampling_does_not_exceed_available_rows(self, gbm, corpus):
        features, _ = corpus
        assert global_importance(gbm, features, k=5, sample=10_000)


class TestUnexplainableModels:
    def test_distilbert_is_refused_with_a_reason(self):
        """The ceiling check has no attribution path, and says so rather than guessing."""
        with pytest.raises(ExplanationError, match="ceiling check"):
            contribution_matrix(
                DistilBertClassifier(),
                portable_features([ScoreRequest(title="x", description="y")]),
            )

    def test_regime_mismatch_is_caught_before_explaining(self, gbm):
        """A FULL FeatureSet through a PORTABLE model must not silently misalign."""
        from veridyx.features import FULL_FEATURE_NAMES, FeatureSet

        wrong = FeatureSet(
            matrix=np.zeros((1, len(FULL_FEATURE_NAMES))),
            names=FULL_FEATURE_NAMES,
            texts=["anything"],
            regime="full",
        )
        assert gbm.regime == PORTABLE
        with pytest.raises(ValueError, match="not interchangeable"):
            contribution_matrix(gbm, wrong)
