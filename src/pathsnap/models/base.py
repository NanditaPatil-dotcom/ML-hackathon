"""Shared feature schema and model registry for candidate-segment scoring.

Schema matches the real preprocessing pipeline's output
(`pathsnap.data.preprocessing.pipeline`, candidates parquet) exactly, so no renaming
is needed between preprocessing and modeling. `candidate_arc_id` itself is
deliberately NOT a model feature: it's a per-trip row index into that trip's
own .arcs file, so the same id means a different road in a different trip —
used as a categorical it would overfit to the training trips and not
generalize to evaluation trips. `candidate_rank` (added in
`pathsnap.data.build_features.add_candidate_rank`) is the trip-independent
substitute.
"""
from dataclasses import dataclass
from typing import Callable

import pandas as pd

NUMERIC_FEATURES = [
    "dist_to_segment_m",
    "bearing_alignment_deg",
    "speed_limit_proxy_mps",
    "point_speed_mps",
    "arc_length_m",
    "candidate_rank",
]
LABEL = "is_ground_truth_arc"
GROUP_COL = "trip_id"     # trip-level grouping, so CV/holdout never splits a trip
POINT_COL = "point_idx"   # groups one GPS point's candidates (top-1 acc, HMM emission norm)
REQUIRED_COLUMNS = NUMERIC_FEATURES + [LABEL]


@dataclass
class ModelSpec:
    """One entry in the model registry: how to build it. Every model family
    here takes the same plain numeric feature matrix — no per-model data prep
    is needed since we dropped the one feature (candidate_arc_id) that would
    have needed special categorical handling."""
    name: str
    build: Callable[[], object]

    def make_X(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[NUMERIC_FEATURES]

    def fit(self, model, X, y):
        model.fit(X, y)
        return model


MODEL_REGISTRY: dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> None:
    MODEL_REGISTRY[spec.name] = spec


def build_models(names: list[str]) -> dict[str, object]:
    return {name: MODEL_REGISTRY[name].build() for name in names}
