"""CatBoost candidate-scoring model.

CatBoost takes `candidate_road_id` as a native categorical feature (passed
via `cat_features=` at fit time, see `ModelSpec.fit`/`uses_native_categorical`
in base.py) rather than needing a hand-rolled encoding. That, plus gradient
boosting on ordered statistics, is why it tends to handle a high-cardinality
id column well without overfitting to rare road ids.
"""
import yaml
from catboost import CatBoostClassifier

from pathsnap.models.base import ModelSpec, register

_DEFAULTS = dict(iterations=500, depth=6, learning_rate=0.05, loss_function="Logloss", random_seed=42, verbose=0)


def load_params(config_path: str | None = None) -> dict:
    params = dict(_DEFAULTS)
    if config_path:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        params.update(cfg.get("catboost", {}))
    return params


def build(config_path: str | None = None) -> CatBoostClassifier:
    return CatBoostClassifier(**load_params(config_path))


register(ModelSpec(name="CatBoost", build=build, uses_native_categorical=True))
