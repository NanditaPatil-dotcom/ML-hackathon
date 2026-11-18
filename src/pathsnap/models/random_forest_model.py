"""Random Forest candidate-scoring model (registry entry).

Hyperparameters follow the Random Forest baseline in
`pathsnap/random_forest.py`; this file exposes the same estimator through the
shared registry so it is trained on the shared feature schema
(`pathsnap.models.base.NUMERIC_FEATURES`) and compared on the same
trip-grouped folds as every other model.
"""
import yaml
from sklearn.ensemble import RandomForestClassifier

from pathsnap.models.base import ModelSpec, register

_DEFAULTS = dict(
    n_estimators=400,
    max_features="sqrt",
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)


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
