"""Extra Trees candidate-scoring model registered in the shared model suite."""
from sklearn.ensemble import ExtraTreesClassifier

from pathsnap.models.base import ModelSpec, load_model_params, register

_DEFAULTS = dict(
    n_estimators=400,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)


def load_params(config_path: str | None = None) -> dict:
    return load_model_params("extra_trees", _DEFAULTS, config_path)


def build(config_path: str | None = None) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(**load_params(config_path))


register(ModelSpec(name="ExtraTrees", build=build))
