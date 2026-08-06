"""LightGBM over TF-IDF terms plus the hand-crafted metadata features.

The best-performing model in the comparison, and the easiest to explain: TreeExplainer
gives exact SHAP values in reasonable time, so every flag this model raises comes with
a defensible per-posting reason.

**On the vocabulary size.** This started at 2,000 terms, on the reasoning that trees
handle wide sparse text poorly and that SHAP over a large vocabulary is unreadable.
Measured on the portable/grouped cell, that reasoning cost real accuracy:

    max_features=2,000   PR-AUC 0.7271
    max_features=10,000  PR-AUC 0.7529
    max_features=30,000  PR-AUC 0.7727

At 2,000 the GBM lost to TF-IDF + logistic regression (0.7584); at 30,000 it wins. The
legibility argument was also simply wrong: TreeExplainer's cost scales with tree
structure, not with vocabulary size, and per-posting SHAP contributions are sparse, so
the top-k terms driving a single decision are just as readable either way. Capping the
vocabulary bought nothing and cost 0.046 PR-AUC.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from veridyx.features import FeatureSet
from veridyx.models.base import Model

TFIDF_PREFIX = "term:"


class GradientBoosting(Model):
    name = "lightgbm"

    def __init__(
        self,
        max_features: int = 30_000,
        n_estimators: int = 400,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=5,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
        )
        self.classifier = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            # Reweight rather than resample, for the same reason as the baseline:
            # duplicating minority rows re-creates the leakage clustering removes.
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        self.feature_names: list[str] = []

    def _design_matrix(self, features: FeatureSet, fit: bool) -> sparse.csr_matrix:
        text_matrix = (
            self.vectorizer.fit_transform(features.texts)
            if fit
            else self.vectorizer.transform(features.texts)
        )
        return sparse.hstack(
            [text_matrix, sparse.csr_matrix(features.matrix)], format="csr"
        )

    def fit(self, features: FeatureSet, labels: np.ndarray) -> GradientBoosting:
        matrix = self._design_matrix(features, fit=True)
        self.feature_names = [
            f"{TFIDF_PREFIX}{t}" for t in self.vectorizer.get_feature_names_out()
        ] + list(features.names)
        self.classifier.fit(matrix, labels.astype(int))
        self.regime = features.regime
        self._fitted = True
        return self

    def predict_proba(self, features: FeatureSet) -> np.ndarray:
        self._check_regime(features)
        return self.classifier.predict_proba(self._design_matrix(features, fit=False))[:, 1]

    def design_matrix(self, features: FeatureSet) -> sparse.csr_matrix:
        """Exposed for the SHAP explainer, which needs the same columns the model saw."""
        self._check_regime(features)
        return self._design_matrix(features, fit=False)
