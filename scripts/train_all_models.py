#!/usr/bin/env python3
"""Train and compare the candidate-scoring models, pick and save the best one.

    python scripts/train_all_models.py --data data/processed/candidates_mock.csv
    python scripts/train_all_models.py --data path/to/real_preprocessed.parquet

Outputs:
    results/metrics/model_comparison.csv  / .md   metrics for every model run
    artifacts/models/<best_model>_best.joblib      chosen model (CandidateScorer)

RBF SVM doesn't scale to a dataset the size of the real candidates table
(roughly O(n^2)-O(n^3) in rows), so the *comparison* runs on a
`--max-points` subsample (whole GPS points, candidates kept together) by
default. CatBoost (gradient boosting, ~linear in rows) is then also fit on
the FULL dataset and saved as the production model if it's the one
selected, so the shipped model isn't trained on a subsample even though the
comparison that picked it was.
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
from pathsnap.evaluation.compare_models import load_dataset, render_markdown, run_comparison, subsample_by_point  # noqa: E402
from pathsnap.models.base import LABEL, MODEL_REGISTRY  # noqa: E402

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
    ap.add_argument("--max-points", type=int, default=cfg.get("max_points_for_comparison", 6000),
                    help="subsample this many GPS points (all their candidates kept) for the "
                         "cross-model comparison, since RBF SVM doesn't scale to the full table")
    ap.add_argument("--refit-best-on-full-data", action="store_true", default=True)
    a = ap.parse_args()

    artifacts_dir, results_dir = Path(a.artifacts_dir), Path(a.results_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_names = [m.strip() for m in a.models.split(",")]
    full_df = load_dataset(a.data)
    print(f"{len(full_df)} rows (full), positive rate {full_df[LABEL].mean():.3f}, models: {model_names}")

    cmp_df = subsample_by_point(full_df, a.max_points)
    if len(cmp_df) < len(full_df):
        print(f"comparison running on a {len(cmp_df)}-row subsample ({a.max_points} GPS points) "
              f"so RBF SVM finishes in reasonable time")

    report, scorers = run_comparison(cmp_df, model_names, folds=a.folds, select_by=a.select_by)

    table = report["cv_mean"].add_prefix("cv_").join(
        report["cv_std"][[a.select_by, "f1", "top1_point_acc"]].add_prefix("cv_std_")
    ).join(report["test"].add_prefix("test_")).loc[report["ranked"].index]
    table.to_csv(results_dir / "model_comparison.csv")
    (results_dir / "model_comparison.md").write_text(render_markdown(report, a.data))
    (results_dir / "summary.json").write_text(json.dumps({
        "selected": report["best"], "select_by": a.select_by, "tied": report["tied"],
        "ranking": report["ranked"].round(5).to_dict(),
        "comparison_rows": len(cmp_df), "full_rows": len(full_df),
    }, indent=2))

    best_name = report["best"]
    best_scorer = scorers[best_name]
    if a.refit_best_on_full_data and len(cmp_df) < len(full_df):
        print(f"refitting {best_name} on the full {len(full_df)}-row dataset for the production model...")
        spec = MODEL_REGISTRY[best_name]
        model = spec.build()
        spec.fit(model, spec.make_X(full_df), full_df[LABEL])
        from pathsnap.models.scorer import CandidateScorer
        best_scorer = CandidateScorer(model=model, model_name=best_name)

    joblib.dump(best_scorer, artifacts_dir / f"{best_name.lower()}_best.joblib")
    print(render_markdown(report, a.data))
    print(f"\nSaved best model ({best_name}) to {artifacts_dir / f'{best_name.lower()}_best.joblib'}")


if __name__ == "__main__":
    main()
