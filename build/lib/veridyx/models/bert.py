"""DistilBERT fine-tune — the ceiling check.

Its job in this project is to answer one question: how much does a pretrained
transformer buy over TF-IDF on this task, once duplicate leakage is removed and the
benchmark-only columns are taken away? If the answer is "not much", that is the
result, and it belongs in the deck rather than in a drawer.

Three choices worth defending:

* **A hand-written training loop, not `Trainer`.** The loop is forty lines, it makes
  the class-weighted loss explicit rather than buried in a config, and it does not
  drag in `accelerate`. For a run this small the abstraction costs more than it saves.

* **Weighted loss, not oversampling.** Same reasoning as the other two models:
  duplicating minority rows would reintroduce exactly the near-duplicate leakage that
  the grouped split exists to remove.

* **256 tokens.** EMSCAD descriptions run to several thousand characters, so this
  truncates most of them. That is a real limitation and it is reported rather than
  hidden — but scam signals cluster heavily in the opening lines (the title, the
  pitch, the money), and 512 tokens doubles the run time for a run that has to fit
  inside a week.

Import is deliberately lazy at module scope: torch is a ~1 GB dependency that the
baseline, the GBM, the endpoint and the quantyx bridge all run without.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from veridyx.features import FeatureSet
from veridyx.models.base import ARTIFACT_DIR, Model

log = logging.getLogger(__name__)

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256


def _torch():
    """Import torch on demand, with an actionable message if it is absent."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment, not logic
        raise ImportError(
            "DistilBERT requires torch. Install it with:\n"
            "  .venv/bin/pip install -r requirements-bert.txt"
        ) from exc
    return torch


def best_device() -> str:
    """MPS on Apple silicon, CUDA where present, CPU otherwise."""
    torch = _torch()
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class DistilBertClassifier(Model):
    name = "distilbert"

    def __init__(
        self,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        max_length: int = MAX_LENGTH,
        device: str | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.seed = seed
        self._device = device
        self._model = None
        self._tokenizer = None

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = best_device()
        return self._device

    def _build(self) -> None:
        torch = _torch()
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2
        ).to(self.device)

    def _batches(self, n: int, shuffle: bool):
        rng = np.random.default_rng(self.seed)
        idx = np.arange(n)
        if shuffle:
            rng.shuffle(idx)
        for start in range(0, n, self.batch_size):
            yield idx[start : start + self.batch_size]

    def _encode(self, texts: list[str]):
        return self._tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

    def fit(self, features: FeatureSet, labels: np.ndarray) -> DistilBertClassifier:
        torch = _torch()
        self._build()

        y = labels.astype(np.int64)
        # Inverse-frequency weights. With ~5% positives this puts roughly 20x the loss
        # on a missed fraud than on a false alarm, which is the right default given
        # that the threshold — not the loss — is where the cost trade-off is actually
        # made later, in veridyx.threshold.
        counts = np.bincount(y, minlength=2).astype(np.float64)
        weights = torch.tensor(
            (counts.sum() / (2.0 * np.maximum(counts, 1.0))),
            dtype=torch.float32,
            device=self.device,
        )
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=self.learning_rate)

        self._model.train()
        n = len(features.texts)
        for epoch in range(self.epochs):
            total, seen = 0.0, 0
            for batch_idx in self._batches(n, shuffle=True):
                batch_texts = [features.texts[i] for i in batch_idx]
                encoded = self._encode(batch_texts)
                targets = torch.tensor(y[batch_idx], device=self.device)

                optimizer.zero_grad()
                logits = self._model(**encoded).logits
                loss = loss_fn(logits, targets)
                loss.backward()
                optimizer.step()

                total += float(loss.item()) * len(batch_idx)
                seen += len(batch_idx)
            log.info("epoch %d/%d  loss %.4f", epoch + 1, self.epochs, total / max(seen, 1))

        self.regime = features.regime
        self._fitted = True
        return self

    def predict_proba(self, features: FeatureSet) -> np.ndarray:
        torch = _torch()
        self._check_regime(features)
        self._model.eval()

        out = np.zeros(len(features.texts), dtype=np.float64)
        with torch.no_grad():
            for batch_idx in self._batches(len(features.texts), shuffle=False):
                encoded = self._encode([features.texts[i] for i in batch_idx])
                logits = self._model(**encoded).logits
                probs = torch.softmax(logits, dim=-1)[:, 1]
                out[batch_idx] = probs.detach().cpu().numpy()
        return out

    # -- persistence ---------------------------------------------------------------
    #
    # Overridden because a fine-tuned transformer does not belong in a pickle: the
    # weights are large, and `save_pretrained` produces a directory the Hugging Face
    # Space can load directly without importing this package.

    def save(self, path: Path | None = None) -> Path:
        if not self._fitted:
            raise RuntimeError("model has not been fitted")
        path = path or ARTIFACT_DIR / f"{self.name}-{self.regime}"
        path.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(path)
        self._tokenizer.save_pretrained(path)
        (path / "veridyx.json").write_text(
            f'{{"name": "{self.name}", "regime": "{self.regime}", '
            f'"version": "{self.version}", "max_length": {self.max_length}}}\n'
        )
        return path

    @classmethod
    def load(cls, path: Path, regime: str) -> DistilBertClassifier:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model = cls()
        model._tokenizer = AutoTokenizer.from_pretrained(path)
        model._model = AutoModelForSequenceClassification.from_pretrained(path).to(
            model.device
        )
        model.regime = regime
        model._fitted = True
        return model
