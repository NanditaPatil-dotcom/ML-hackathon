#!/usr/bin/env python3
"""Sanity check for the HMM/API-facing probability interface."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pathsnap.evaluation.compare_models import load_dataset  # noqa: E402
from pathsnap.models.scorer import CandidateScorer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    scorer = CandidateScorer.load(ROOT / "artifacts/models/catboost_best.joblib")
    df = load_dataset(ROOT / "data/processed/candidates_mock.csv").head(20)

    proba = scorer.predict_proba(df)
    emission = scorer.emission_log_prob(df)

    assert ((proba >= 0) & (proba <= 1)).all(), "probabilities out of [0,1]"
    assert len(proba) == len(df)

    print("model:", scorer.model_name)
    print("sample probabilities:", proba[:5].round(4))
    print("sample emission log-probs:", emission[:5].round(4))
    print("OK")


if __name__ == "__main__":
    main()
