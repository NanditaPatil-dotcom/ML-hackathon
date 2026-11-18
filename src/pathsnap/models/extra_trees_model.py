"""Extra Trees candidate-scoring model (registry entry).

Hyperparameters follow `pathsnap.models.extra_tree.ExtraTreeModel`, the
standalone Extra Trees wrapper; this file just exposes the same estimator
through the shared registry so `scripts/train_all_models.py` can train and
compare it with the other model families on the same features and folds.
"""
import yaml
from sklearn.ensemble import ExtraTreesClassifier

from pathsnap.models.base import ModelSpec, register

_DEFAULTS = dict(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)


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
