"""Probability interface for the HMM/Viterbi step.

    from pathsnap.models.scorer import CandidateScorer
    scorer = CandidateScorer.load("artifacts/models/catboost_best.joblib")
    df["p_match"] = scorer.predict_proba(df)          # P(candidate is the true road)
    df["emission"] = scorer.emission_log_prob(df)     # log P, normalised per GPS point

`df` needs the feature columns in `pathsnap.models.base` (label not
required). If a `point_id` column is present, `emission_log_prob` normalises
over each point's candidates so they sum to 1, which is what Viterbi expects
as emission probabilities.
"""
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from pathsnap.models.base import ID_COL, NUMERIC_FEATURES, add_road_freq


@dataclass
class CandidateScorer:
    model: object
    model_name: str
    road_freq: dict = field(default_factory=dict)
    uses_native_categorical: bool = False

    @classmethod
    def load(cls, path) -> "CandidateScorer":
        return joblib.load(path)

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.uses_native_categorical:
            X = df[NUMERIC_FEATURES + [ID_COL]].copy()
            X[ID_COL] = X[ID_COL].astype(str)
            return X
        return add_road_freq(df, self.road_freq)[NUMERIC_FEATURES + ["road_freq"]]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """P(candidate is the true road segment), shape (n_rows,)."""
        return self.model.predict_proba(self._X(df))[:, 1]

    def emission_log_prob(self, df: pd.DataFrame, point_col: str = "point_id", eps: float = 1e-9) -> np.ndarray:
        """Log emission scores; normalised per GPS point when point_col exists."""
        p = self.predict_proba(df)
        if point_col in df.columns:
            s = pd.Series(p, index=df.index).groupby(df[point_col]).transform("sum").values
            p = p / np.maximum(s, eps)
        return np.log(np.clip(p, eps, 1.0))
