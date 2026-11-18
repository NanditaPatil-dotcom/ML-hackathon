"""Sanity check for the HMM-facing probability interface."""
import pandas as pd
from candidate_scorer import CandidateScorer

scorer = CandidateScorer.load("../outputs/best_model.joblib")
df = pd.read_csv("../data/mock/candidates_mock.csv").head(20)

proba = scorer.predict_proba(df)
emission = scorer.emission_log_prob(df)

assert ((proba >= 0) & (proba <= 1)).all(), "probabilities out of [0,1]"
assert len(proba) == len(df)

per_point_sum = pd.Series(pd.Series(proba).values).groupby(df["point_id"].values).apply(lambda s: s.sum())
print("model:", scorer.model_name)
print("sample probabilities:", proba[:5].round(4))
print("sample emission log-probs:", emission[:5].round(4))
print("OK")
