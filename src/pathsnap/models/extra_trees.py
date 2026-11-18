"""Extra Trees candidate-scoring model.

Like Random Forest but each tree's splits are chosen randomly (not by the
best split search) rather than built from a bootstrap resample — usually a
bit faster to train and slightly more regularized. Takes the same shared
numeric features from pathsnap.models.base as every other model here
(dist_to_segment_m, bearing_alignment_deg, speed_limit_proxy_mps,
point_speed_mps, arc_length_m, candidate_rank).
"""
import yaml
from sklearn.ensemble import ExtraTreesClassifier

from pathsnap.models.base import ModelSpec, register

_DEFAULTS = dict(n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=42)


def load_params(config_path: str | None = None) -> dict:
    params = dict(_DEFAULTS)
    if config_path:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        params.update(cfg.get("extra_trees", {}))
    return params


def build(config_path: str | None = None) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(**load_params(config_path))


register(ModelSpec(name="ExtraTrees", build=build))
