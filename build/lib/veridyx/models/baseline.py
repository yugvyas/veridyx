"""TF-IDF + logistic regression.

The baseline, and not a formality. It trains in seconds, its coefficients are
directly readable, and on text-classification problems of this size it is frequently
competitive with far heavier machinery. If DistilBERT does not clearly beat this, the
honest conclusion is that the extra 66M parameters bought nothing here — and that is
a result worth reporting, not one worth burying.

`class_weight="balanced"` rather than resampling: with ~5% positives, oversampling the
minority duplicates the very scam-campaign rows that near-duplicate clustering exists
to control, which would quietly reintroduce the leakage through the back door.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from veridyx.features import FeatureSet
from veridyx.models.base import Model


class TfidfLogisticRegression(Model):
    name = "tfidf-lr"

    def __init__(
        self,
        max_features: int = 50_000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        C: float = 1.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            # Sublinear scaling matters here: a scam that repeats "money" forty times
            # should not score forty times more suspicious than one that says it once.
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
        )
        self.classifier = LogisticRegression(
            C=C,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
            solver="liblinear",
        )

    def fit(self, features: FeatureSet, labels: np.ndarray) -> TfidfLogisticRegression:
        matrix = self.vectorizer.fit_transform(features.texts)
        self.classifier.fit(matrix, labels.astype(int))
        self.regime = features.regime
        self._fitted = True
        return self

    def predict_proba(self, features: FeatureSet) -> np.ndarray:
        self._check_regime(features)
        matrix = self.vectorizer.transform(features.texts)
        return self.classifier.predict_proba(matrix)[:, 1]

    def top_terms(self, k: int = 25) -> list[tuple[str, float]]:
        """Largest positive coefficients — the words that push toward 'fraud'.

        For a linear model this *is* the global explanation; SHAP on a linear model
        returns coefficient * (value - mean), so these terms are the same story told
        without the extra machinery.
        """
        if not self._fitted:
            raise RuntimeError("model has not been fitted")
        vocab = np.array(self.vectorizer.get_feature_names_out())
        coefs = self.classifier.coef_[0]
        order = np.argsort(coefs)[::-1][:k]
        return [(str(vocab[i]), float(coefs[i])) for i in order]
