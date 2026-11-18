#!/usr/bin/env python3
"""Train and evaluate LightGBM candidate scoring model.
Model Owner: Wingsfry (LightGBM Lead)

Usage:
    python scripts/experiments/train_lgbm.py --data data/processed/candidates_mock.csv
    python scripts/experiments/train_lgbm.py --data results/preprocessing/candidate_data/train_candidates.parquet
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pathsnap.models.lightgbm_model  # noqa: E402
from pathsnap.evaluation.compare_models import load_dataset, subsample_by_point
from pathsnap.models.base import LABEL, MODEL_REGISTRY, NUMERIC_FEATURES, POINT_COL
from pathsnap.models.scorer import CandidateScorer

def top1_accuracy(df, proba):
    if POINT_COL not in df.columns:
        return np.nan
    tmp = df[[POINT_COL, LABEL]].copy()
    tmp["p"] = proba
    best = tmp.loc[tmp.groupby(POINT_COL)["p"].idxmax()]
    return float(best[LABEL].mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/mock/candidates_mock.csv")
    ap.add_argument("--out-dir", default="results/metrics")
    ap.add_argument("--artifacts-dir", default="artifacts/models")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    artifacts_dir = Path(args.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Wingsfry - LightGBM] Loading dataset from: {args.data}")
    df = load_dataset(args.data)
    print(f"Loaded {len(df):,} candidate rows. Positive rate: {df[LABEL].mean():.3f}")

    spec = MODEL_REGISTRY["LightGBM"]
    X = spec.make_X(df)
    y = df[LABEL].values
    groups = df["trip_id"].values if "trip_id" in df.columns else df[POINT_COL].values

    print(f"Features: {list(X.columns)}")
    print(f"Running {args.folds}-Fold GroupKFold Cross-Validation...")

    gkf = GroupKFold(n_splits=args.folds)
    cv_loglosses, cv_top1s, cv_f1s, cv_aucs = [], [], [], []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        df_val = df.iloc[val_idx]

        model = spec.build()
        spec.fit(model, X_tr, y_tr)

        proba = model.predict_proba(X_val)[:, 1]
        pred = (proba >= 0.5).astype(int)

        cv_loglosses.append(log_loss(y_val, np.clip(proba, 1e-7, 1 - 1e-7)))
        cv_top1s.append(top1_accuracy(df_val, proba))
        cv_f1s.append(f1_score(y_val, pred, average="macro"))
        cv_aucs.append(roc_auc_score(y_val, proba))

    print(f"\n[LightGBM - Wingsfry] Cross-Validation Results:")
    print(f"  CV Log Loss:     {np.mean(cv_loglosses):.4f} ± {np.std(cv_loglosses):.4f}")
    print(f"  CV Top-1 Acc:    {np.mean(cv_top1s)*100:.2f}%")
    print(f"  CV Macro F1:     {np.mean(cv_f1s):.4f}")
    print(f"  CV ROC AUC:      {np.mean(cv_aucs):.4f}")

    # Fit on full data
    print("Fitting production LightGBM model on full dataset...")
    t0 = time.time()
    final_model = spec.build()
    spec.fit(final_model, X, y)
    train_time = round(time.time() - t0, 3)

    scorer = CandidateScorer(model=final_model, model_name="LightGBM")
    model_path = artifacts_dir / "lightgbm_best.joblib"
    joblib.dump(scorer, model_path)
    print(f"Saved model artifact to: {model_path}")

    metrics = {
        "model_name": "LightGBM",
        "contributor": "Wingsfry",
        "cv_log_loss": round(float(np.mean(cv_loglosses)), 4),
        "cv_log_loss_std": round(float(np.std(cv_loglosses)), 4),
        "cv_top1_acc": round(float(np.mean(cv_top1s)), 4),
        "cv_macro_f1": round(float(np.mean(cv_f1s)), 4),
        "cv_roc_auc": round(float(np.mean(cv_aucs)), 4),
        "train_time_sec": train_time
    }

    metrics_path = out_dir / "lightgbm_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics summary to: {metrics_path}")

if __name__ == "__main__":
    main()
