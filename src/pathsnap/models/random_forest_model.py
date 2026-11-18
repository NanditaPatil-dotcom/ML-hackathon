"""Random Forest candidate-scoring model (registry entry).

This estimator uses the shared feature schema
(`pathsnap.models.base.NUMERIC_FEATURES`) and compared on the same
trip-grouped folds as every other model.
"""
from sklearn.ensemble import RandomForestClassifier

from pathsnap.models.base import ModelSpec, load_model_params, register

_DEFAULTS = dict(
    n_estimators=400,
    max_features="sqrt",
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)


def load_params(config_path: str | None = None) -> dict:
    return load_model_params("random_forest", _DEFAULTS, config_path)


def build(config_path: str | None = None) -> RandomForestClassifier:
    return RandomForestClassifier(**load_params(config_path))


register(ModelSpec(name="RandomForest", build=build))
