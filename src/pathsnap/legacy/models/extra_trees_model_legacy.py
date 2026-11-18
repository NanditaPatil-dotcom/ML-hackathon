"""Compatibility import for the former ExtraTrees registry module.

Kept for old notebooks; the canonical implementation is
``pathsnap.models.extra_trees``. Importing this module no longer registers a
second model with the same registry name.
"""
from pathsnap.models.extra_trees import build, load_params

__all__ = ["build", "load_params"]
