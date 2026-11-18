#!/usr/bin/env python3
"""Evaluate an already-trained CandidateScorer against a (new) dataset.

Useful once real preprocessed data lands: check how a model trained on mock
data holds up, without retraining.

    python scripts/evaluate_models.py --model artifacts/models/catboost_best.joblib --data data/processed/real.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pathsnap.evaluation.compare_models import load_dataset  # noqa: E402
from pathsnap.evaluation.metrics import compute_metrics  # noqa: E402
from pathsnap.models.scorer import CandidateScorer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    a = ap.parse_args()

    scorer = CandidateScorer.load(a.model)
    df = load_dataset(a.data)
    proba = scorer.predict_proba(df)
    metrics = compute_metrics(df["label"].values, proba, df)

    print(f"model: {scorer.model_name}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
