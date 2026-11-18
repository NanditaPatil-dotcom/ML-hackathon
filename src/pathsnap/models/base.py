"""Shared feature schema and per-model data prep for candidate-segment scoring.

Every model family needs the same 5 raw columns but turns them into features
differently (CatBoost/LightGBM take the road ID as a native categorical;
everything else needs a numeric encoding instead). This module is the one
place that knows the raw schema and dispatches the per-model prep, so
`models/catboost_model.py`, `models/rbf_svm.py`, and any teammate's new
`models/<family>.py` all plug into `evaluation/compare_models.py` the same way.
"""
from dataclasses import dataclass
from typing import Callable

import pandas as pd

# The feature contract agreed with the preprocessing team. Any CSV/Parquet
# fed into this package must have these columns (plus optionally point_id /
# trip_id for grouped splits and per-point top-1 accuracy).
NUMERIC_FEATURES = ["distance_to_segment", "bearing_diff", "speed_limit", "transition_distance"]
ID_COL = "candidate_road_id"
LABEL = "label"
REQUIRED_COLUMNS = NUMERIC_FEATURES + [ID_COL, LABEL]


def fit_road_freq(df: pd.DataFrame) -> dict:
    """Frequency-encode the road id for models that can't take it as a raw
    categorical (an integer id has no meaningful order)."""
    return df[ID_COL].value_counts(normalize=True).to_dict()


def add_road_freq(df: pd.DataFrame, freq_map: dict) -> pd.DataFrame:
    out = df.copy()
    out["road_freq"] = out[ID_COL].map(freq_map).fillna(0.0)
    return out


@dataclass
class ModelSpec:
    """One entry in the model registry: how to build it, prep X for it, and
    fit it. `uses_native_categorical=True` means it takes `candidate_road_id`
    directly (as a string/category) instead of the frequency encoding."""
    name: str
    build: Callable[[], object]
    uses_native_categorical: bool = False

    def make_X(self, df: pd.DataFrame, freq_map: dict) -> pd.DataFrame:
        if self.uses_native_categorical:
            X = df[NUMERIC_FEATURES + [ID_COL]].copy()
            X[ID_COL] = X[ID_COL].astype(str)
            return X
        return add_road_freq(df, freq_map)[NUMERIC_FEATURES + ["road_freq"]]

    def fit(self, model, X, y):
        if self.uses_native_categorical:
            model.fit(X, y, cat_features=[ID_COL])
        else:
            model.fit(X, y)
        return model


MODEL_REGISTRY: dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> None:
    MODEL_REGISTRY[spec.name] = spec


def build_models(names: list[str]) -> dict[str, object]:
    return {name: MODEL_REGISTRY[name].build() for name in names}
