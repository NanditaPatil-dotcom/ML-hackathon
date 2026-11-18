"""Random Forest candidate-scoring model.

Bootstrap-aggregated decision trees. Takes the same shared numeric features
from pathsnap.models.base as every other model here (dist_to_segment_m,
bearing_alignment_deg, speed_limit_proxy_mps, point_speed_mps,
arc_length_m, candidate_rank).

(There's also a standalone src/pathsnap/random_forest.py with its own
feature engineering straight from the raw Student folder files — a
different, earlier approach. This module is the one registered into
pathsnap.models.base.MODEL_REGISTRY for the shared comparison pipeline.)
"""
import yaml
from sklearn.ensemble import RandomForestClassifier

from pathsnap.models.base import ModelSpec, register

_DEFAULTS = dict(n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=42)


def load_params(config_path: str | None = None) -> dict:
    params = dict(_DEFAULTS)
    if config_path:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        params.update(cfg.get("random_forest", {}))
    return params


def build(config_path: str | None = None) -> RandomForestClassifier:
    return RandomForestClassifier(**load_params(config_path))


register(ModelSpec(name="RandomForest", build=build))
