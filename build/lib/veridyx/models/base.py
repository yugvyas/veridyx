"""The interface every Veridyx model implements, plus persistence.

Keeping this narrow is what makes the results table trustworthy. Three architectures
that each define their own metric code, their own split handling and their own
threshold would produce three numbers that cannot honestly be placed in one column.
Here they differ in exactly one respect — `fit` and `predict_proba` — and everything
around them is shared.
"""

from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from veridyx.features import FeatureSet

ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACT_DIR = ROOT / "artifacts"


class Model(ABC):
    """A fraud scorer.

    `name` identifies the architecture; `version` identifies a trained instance and is
    recorded on every `Verdict`, so a flag reviewed last week can always be traced to
    the exact model that raised it.
    """

    name: str = "abstract"

    def __init__(self) -> None:
        self.regime: str | None = None
        self._fitted = False

    @property
    def version(self) -> str:
        return f"{self.name}-{self.regime or 'unfit'}-v{self.schema_version}"

    schema_version: int = 1

    @abstractmethod
    def fit(self, features: FeatureSet, labels: np.ndarray) -> Model: ...

    @abstractmethod
    def predict_proba(self, features: FeatureSet) -> np.ndarray:
        """Probability of fraud, shape (n,), in [0, 1]."""

    def _check_regime(self, features: FeatureSet) -> None:
        """Refuse to score a regime the model was not trained on.

        Without this, a FULL-trained model silently accepts a PORTABLE FeatureSet
        whose matrix has fewer columns, and either crashes deep inside a library or —
        worse — succeeds against a misaligned column order and returns plausible
        garbage. This is the single most likely way the two-regime design could
        produce a wrong number that nobody notices.
        """
        if not self._fitted:
            raise RuntimeError(f"{self.name} has not been fitted")
        if self.regime != features.regime:
            raise ValueError(
                f"{self.name} was trained on regime {self.regime!r} but was given "
                f"{features.regime!r}. These have different feature sets and are not "
                "interchangeable."
            )

    # -- persistence ---------------------------------------------------------------

    def save(self, path: Path | None = None) -> Path:
        path = path or ARTIFACT_DIR / f"{self.name}-{self.regime}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh)
        meta = path.with_suffix(".json")
        meta.write_text(
            json.dumps(
                {"name": self.name, "regime": self.regime, "version": self.version},
                indent=2,
            )
            + "\n"
        )
        return path


def load_model(path: Path) -> Model:
    with Path(path).open("rb") as fh:
        model = pickle.load(fh)
    if not isinstance(model, Model):
        raise TypeError(f"{path} does not contain a Veridyx model")
    return model
