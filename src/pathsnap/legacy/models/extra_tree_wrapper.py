"""Extra Trees candidate scorer for PathSnap map matching.

Rows should represent observation/road-segment candidate pairs. ``y`` should
mark whether each candidate is the reference road segment (1) or not (0).
The returned positive-class scores can be used as emissions by the HMM decoder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from sklearn.ensemble import ExtraTreesClassifier
except ImportError as exc:  # pragma: no cover - depends on environment setup
    raise ImportError(
        "ExtraTreeModel requires scikit-learn. Install it with: pip install scikit-learn"
    ) from exc


class ExtraTreeModel:
    """Train and score candidate road segments with Extra Trees.

    This wrapper deliberately leaves feature construction and data splitting to
    the shared pipeline. It accepts array-like inputs or pandas DataFrames.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 300,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | float | int | None = "sqrt",
        class_weight: str | dict[int, float] | None = "balanced",
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.estimator = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    @property
    def is_fitted(self) -> bool:
        """Whether the underlying classifier has been fitted."""
        return hasattr(self.estimator, "classes_")

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Any | None = None,
    ) -> "ExtraTreeModel":
        """Fit on candidate features and binary match labels."""
        self.estimator.fit(X, y, sample_weight=sample_weight)
        if 1 not in self.estimator.classes_:
            raise ValueError("Training labels must include positive class 1.")
        return self

    def predict_candidate_scores(self, X: Any) -> Any:
        """Return estimated probability that each candidate is the true match."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit(X, y) first.")
        positive_index = list(self.estimator.classes_).index(1)
        return self.estimator.predict_proba(X)[:, positive_index]

    def predict(self, X: Any) -> Any:
        """Return binary candidate labels (1=match, 0=non-match)."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit(X, y) first.")
        return self.estimator.predict(X)

    def save(self, path: str | Path) -> None:
        """Save the fitted wrapper with joblib."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")
        try:
            import joblib
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError("Saving requires joblib (normally installed with scikit-learn).") from exc

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    @classmethod
    def load(cls, path: str | Path) -> "ExtraTreeModel":
        """Load a model previously written with :meth:`save`."""
        try:
            import joblib
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError("Loading requires joblib (normally installed with scikit-learn).") from exc

        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Expected a {cls.__name__} artifact, got {type(model).__name__}.")
        return model
