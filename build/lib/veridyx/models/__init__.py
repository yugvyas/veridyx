"""The three architectures, behind one interface.

Every model takes a `FeatureSet` and returns calibrated-ish probabilities. They are
interchangeable by construction so that the comparison in the results table is
genuinely like-for-like: same splits, same regime, same metric code, same threshold
machinery. The only thing that varies is the architecture.
"""

from veridyx.models.base import Model, load_model

__all__ = ["Model", "load_model"]
