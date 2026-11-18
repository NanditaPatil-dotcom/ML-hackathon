"""RBF-kernel SVM candidate-scoring model.

Every feature is standardized first (`StandardScaler`) since SVMs are
scale-sensitive (distances in meters, degrees, m/s all on different scales).
`probability=True` fits Platt scaling on top of the decision function so
`predict_proba` returns a calibrated-ish score, which is what the HMM
emission step needs.
"""
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pathsnap.models.base import ModelSpec, load_model_params, register

_DEFAULTS = dict(kernel="rbf", C=3.0, gamma="scale", probability=True, random_state=42)


def load_params(config_path: str | None = None) -> dict:
    return load_model_params("rbf_svm", _DEFAULTS, config_path)


def build(config_path: str | None = None):
    return make_pipeline(StandardScaler(), SVC(**load_params(config_path)))


register(ModelSpec(name="RBF_SVM", build=build))
