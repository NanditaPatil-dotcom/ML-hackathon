"""Trip-level train/validation split.

The raw dataset already partitions trips into train / student_test /
evaluator_test directories, so no re-split across those boundaries is
needed or permitted (see Dataset Exploration guidelines on leakage
prevention). This module only adds an optional k-fold split *within* the
80 training trips, by trip id, for cross-validation -- never by individual
GPS point, which would leak a trip's points across folds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def kfold_by_trip(trip_ids: list[str], n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """Return a DataFrame with columns [trip_id, fold] assigning each trip to
    exactly one of ``n_folds`` folds, shuffled with a fixed seed for
    reproducibility.
    """
    rng = np.random.default_rng(seed)
    ids = list(trip_ids)
    rng.shuffle(ids)
    folds = np.arange(len(ids)) % n_folds
    return pd.DataFrame({"trip_id": ids, "fold": folds}).sort_values("trip_id").reset_index(
        drop=True
    )
