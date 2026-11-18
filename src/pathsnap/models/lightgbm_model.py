"""LightGBM candidate-scoring model.
Model Owner: Wingsfry (LightGBM Lead)

Gradient-boosted decision trees using LightGBM for scoring candidate road segments.
Takes the shared numeric features from pathsnap.models.base:
    dist_to_segment_m, bearing_alignment_deg, speed_limit_proxy_mps,
    point_speed_mps, arc_length_m, candidate_rank
"""
import yaml
from lightgbm import LGBMClassifier

from pathsnap.models.base import ModelSpec, register

_DEFAULTS = dict(
    n_estimators=150,
    learning_rate=0.08,
    num_leaves=31,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
    n_jobs=-1
)

def load_params(config_path: str | None = None) -> dict:
    params = dict(_DEFAULTS)
    if config_path:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        params.update(cfg.get("lightgbm", {}))
    return params

def build(config_path: str | None = None) -> LGBMClassifier:
    return LGBMClassifier(**load_params(config_path))

register(ModelSpec(name="LightGBM", build=build))
