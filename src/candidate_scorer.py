"""Probability interface for the HMM/Viterbi step.

    from candidate_scorer import CandidateScorer
    scorer = CandidateScorer.load("outputs/best_model.joblib")
    df["p_match"] = scorer.predict_proba(df)          # P(candidate is the true road)
    df["emission"] = scorer.emission_log_prob(df)     # log P, normalised per GPS point

`df` needs the feature columns below (label not required). If a `point_id`
column is present, emission_log_prob normalises over each point's candidates
so they sum to 1, which is what Viterbi expects as emission probabilities.
"""
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

NUMERIC_FEATURES = ["distance_to_segment", "bearing_diff", "speed_limit", "transition_distance"]
ID_COL = "candidate_road_id"
LABEL = "label"


def fit_road_freq(df):
    return (df[ID_COL].value_counts(normalize=True)).to_dict()


def add_road_freq(df, freq_map):
    out = df.copy()
    out["road_freq"] = out[ID_COL].map(freq_map).fillna(0.0)
    return out


@dataclass
class CandidateScorer:
    model: object
    model_name: str
    road_freq: dict = field(default_factory=dict)
    train_categories: list | None = None

    @classmethod
    def load(cls, path):
        return joblib.load(path)

    def _X(self, df):
        if self.model_name == "CatBoost":
            X = df[NUMERIC_FEATURES + [ID_COL]].copy()
            X[ID_COL] = X[ID_COL].astype(str)
            return X
        if self.model_name == "LightGBM":
            X = df[NUMERIC_FEATURES + [ID_COL]].copy()
            X[ID_COL] = pd.Categorical(X[ID_COL], categories=self.train_categories)
            return X
        return add_road_freq(df, self.road_freq)[NUMERIC_FEATURES + ["road_freq"]]

    def predict_proba(self, df) -> np.ndarray:
        """P(candidate is the true road segment), shape (n_rows,)."""
        return self.model.predict_proba(self._X(df))[:, 1]

    def emission_log_prob(self, df, point_col="point_id", eps=1e-9) -> np.ndarray:
        """Log emission scores; normalised per GPS point when point_col exists."""
        p = self.predict_proba(df)
        if point_col in df.columns:
            s = pd.Series(p, index=df.index).groupby(df[point_col]).transform("sum").values
            p = p / np.maximum(s, eps)
        return np.log(np.clip(p, eps, 1.0))
