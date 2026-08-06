"""Veridyx — fraudulent job posting detection.

The package is organised around one idea: the model that gets *evaluated* and the
model that gets *deployed* must be the same object, trained on features that exist
outside the benchmark. See `veridyx.features` for the two regimes that enforce it.
"""

__version__ = "0.1.0"
