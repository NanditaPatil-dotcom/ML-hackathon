"""CatBoost candidate-scoring model.

Plain numeric features only (see base.py) — gradient boosting with ordered
statistics handles this feature set well without needing any categorical
handling, since the one id-like column (candidate_arc_id) isn't used.
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


register(ModelSpec(name="CatBoost", build=build))
