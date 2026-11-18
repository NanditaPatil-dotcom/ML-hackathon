"""Probability interface for the HMM/Viterbi step and the API.

    from pathsnap.models.scorer import CandidateScorer
    scorer = CandidateScorer.load("artifacts/models/catboost_best.joblib")
    df["p_match"] = scorer.predict_proba(df)          # P(candidate is the true road)
    df["emission"] = scorer.emission_log_prob(df)     # log P, normalised per GPS point

`df` must have the columns in `pathsnap.models.base.NUMERIC_FEATURES`,
INCLUDING `candidate_rank` — call
`pathsnap.data.build_features.add_candidate_rank(df)` first if it isn't
already there (it has to be computed over each (trip_id, point_idx) group,
so it can't be derived from a single row). If a `point_idx`/`trip_id`
column is present, `emission_log_prob` normalises over each point's
candidates so they sum to 1, which is what Viterbi expects as emission
probabilities.
"""
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from pathsnap.models.base import GROUP_COL, NUMERIC_FEATURES, POINT_COL


@dataclass
class CandidateScorer:
    model: object
    model_name: str

    @classmethod
    def load(cls, path) -> "CandidateScorer":
        return joblib.load(path)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """P(candidate is the true road segment), shape (n_rows,)."""
        return self.model.predict_proba(df[NUMERIC_FEATURES])[:, 1]

    def emission_log_prob(self, df: pd.DataFrame, eps: float = 1e-9) -> np.ndarray:
        """Log emission scores; normalised per GPS point (grouped by
        trip_id + point_idx together, since point_idx resets per trip) when
        those columns are present."""
        p = self.predict_proba(df)
        if GROUP_COL in df.columns and POINT_COL in df.columns:
            s = pd.Series(p, index=df.index).groupby([df[GROUP_COL], df[POINT_COL]]).transform("sum").values
            p = p / np.maximum(s, eps)
        return np.log(np.clip(p, eps, 1.0))
