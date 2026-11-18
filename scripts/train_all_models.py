#!/usr/bin/env python3
"""Train and compare the candidate-scoring models, pick and save the best one.

    python scripts/train_all_models.py --data data/processed/candidates_mock.csv
    python scripts/train_all_models.py --data path/to/real_preprocessed.csv

Outputs (under --out-dir's results/ and artifacts/ subtrees by default):
    results/metrics/model_comparison.csv  / .md   metrics for every model run
    artifacts/models/<best_model>.joblib           chosen model (CandidateScorer)
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import pathsnap.models.catboost_model  # noqa: E402  (registers "CatBoost")
import pathsnap.models.rbf_svm  # noqa: E402  (registers "RBF_SVM")
from pathsnap.evaluation.compare_models import load_dataset, render_markdown, run_comparison  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())["comparison"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--artifacts-dir", default="artifacts/models")
    ap.add_argument("--results-dir", default="results/metrics")
    ap.add_argument("--folds", type=int, default=cfg["folds"])
    ap.add_argument("--select-by", default=cfg["select_by"])
    ap.add_argument("--models", default=",".join(cfg["models"]))
    a = ap.parse_args()

    artifacts_dir, results_dir = Path(a.artifacts_dir), Path(a.results_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_names = [m.strip() for m in a.models.split(",")]
    df = load_dataset(a.data)
    print(f"{len(df)} rows, positive rate {df['label'].mean():.3f}, models: {model_names}")

    report, scorers = run_comparison(df, model_names, folds=a.folds, select_by=a.select_by)

    table = report["cv_mean"].add_prefix("cv_").join(
        report["cv_std"][[a.select_by, "f1", "top1_point_acc"]].add_prefix("cv_std_")
    ).join(report["test"].add_prefix("test_")).loc[report["ranked"].index]
    table.to_csv(results_dir / "model_comparison.csv")
    (results_dir / "model_comparison.md").write_text(render_markdown(report, a.data))
    (results_dir / "summary.json").write_text(json.dumps({
        "selected": report["best"], "select_by": a.select_by, "tied": report["tied"],
        "ranking": report["ranked"].round(5).to_dict(),
    }, indent=2))

    best_name = report["best"]
    joblib.dump(scorers[best_name], artifacts_dir / f"{best_name.lower()}_best.joblib")
    print(render_markdown(report, a.data))
    print(f"\nSaved best model ({best_name}) to {artifacts_dir / f'{best_name.lower()}_best.joblib'}")


if __name__ == "__main__":
    main()
