"""Train and compare model families for scoring candidate road segments.

The team's full comparison needs 5 model families (CatBoost, LightGBM, Extra
Trees, Random Forest, RBF SVM). This script covers CatBoost and RBF SVM;
the other three are being trained in parallel by other teammates against the
same mock-data interface, and their results slot into the same
model_comparison table/report.

Usage:
    python src/train_compare.py --data data/mock/candidates_mock.csv
    python src/train_compare.py --data path/to/real_preprocessed.csv --out-dir outputs
    python src/train_compare.py --data ... --models CatBoost,RBF_SVM   (default)

Input: a CSV/Parquet with columns
    distance_to_segment, bearing_diff, speed_limit, transition_distance,
    candidate_road_id, label
and optionally point_id / trip_id (used for leak-free grouped splits and for
per-point top-1 accuracy).

Outputs (in --out-dir):
    model_comparison.csv / model_comparison.md   metrics for every model run
    best_model.joblib                            chosen model, load with
                                                 src/candidate_scorer.py
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                             log_loss, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from candidate_scorer import NUMERIC_FEATURES, ID_COL, LABEL, CandidateScorer, add_road_freq, fit_road_freq

SEED = 42

ALL_MODELS = {
    "CatBoost": lambda: CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05, loss_function="Logloss",
        random_seed=SEED, verbose=0),
    "RBF_SVM": lambda: make_pipeline(StandardScaler(), SVC(
        kernel="rbf", C=3.0, gamma="scale", probability=True, random_state=SEED)),
}


def build_models(names):
    return {name: ALL_MODELS[name]() for name in names}


def make_X(df, name, freq_map):
    """CatBoost gets the road ID as a native categorical; other models get a
    frequency encoding instead (a raw integer ID has no order)."""
    if name == "CatBoost":
        X = df[NUMERIC_FEATURES + [ID_COL]].copy()
        X[ID_COL] = X[ID_COL].astype(str)
        return X
    return add_road_freq(df, freq_map)[NUMERIC_FEATURES + ["road_freq"]]


def fit_model(model, name, X, y):
    if name == "CatBoost":
        model.fit(X, y, cat_features=[ID_COL])
    else:
        model.fit(X, y)
    return model


def align_categories(X_tr, X_te, name):
    return X_te


def top1_accuracy(df, proba):
    """Per GPS point: does the highest-scored candidate equal the true road?
    This is what matters for the HMM emission step."""
    if "point_id" not in df.columns:
        return np.nan
    tmp = df[["point_id", LABEL]].copy()
    tmp["p"] = proba
    best = tmp.loc[tmp.groupby("point_id")["p"].idxmax()]
    return float(best[LABEL].mean())


def metrics(y, proba, df, thr=0.5):
    pred = (proba >= thr).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
        "log_loss": log_loss(y, np.clip(proba, 1e-7, 1 - 1e-7)),
        "brier": brier_score_loss(y, proba),
        "top1_point_acc": top1_accuracy(df, proba),
    }


def group_col(df):
    for c in ("trip_id", "point_id"):
        if c in df.columns:
            return c
    return None


def load(path):
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    missing = set(NUMERIC_FEATURES + [ID_COL, LABEL]) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    return df.dropna(subset=NUMERIC_FEATURES + [ID_COL, LABEL]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--select-by", default="log_loss",
                    help="metric used to pick the final model (lower is better for log_loss/brier)")
    ap.add_argument("--models", default="CatBoost,RBF_SVM",
                    help=f"comma-separated subset of {list(ALL_MODELS)}")
    a = ap.parse_args()
    model_names = [m.strip() for m in a.models.split(",")]
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    df = load(a.data)
    g = group_col(df)
    print(f"{len(df)} rows, positive rate {df[LABEL].mean():.3f}, grouped split on: {g}")

    # 80/20 holdout, grouped so candidates of one point/trip never straddle the split
    if g:
        tr_idx, te_idx = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(df, groups=df[g]))
    else:
        tr_idx, te_idx = train_test_split(np.arange(len(df)), test_size=0.2, stratify=df[LABEL], random_state=SEED)
    train, test = df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)

    # ---- k-fold CV on the training split ----
    splitter = GroupKFold(a.folds) if g else StratifiedKFold(a.folds, shuffle=True, random_state=SEED)
    split_args = (train, train[LABEL], train[g]) if g else (train, train[LABEL])
    cv_rows = []
    for fold, (i_tr, i_va) in enumerate(splitter.split(*split_args)):
        f_tr, f_va = train.iloc[i_tr], train.iloc[i_va]
        fmap = fit_road_freq(f_tr)
        for name, model in build_models(model_names).items():
            X_tr, X_va = make_X(f_tr, name, fmap), make_X(f_va, name, fmap)
            X_va = align_categories(X_tr, X_va, name)
            fit_model(model, name, X_tr, f_tr[LABEL])
            p = model.predict_proba(X_va)[:, 1]
            cv_rows.append({"model": name, "fold": fold, **metrics(f_va[LABEL].values, p, f_va.reset_index(drop=True))})
        print(f"fold {fold + 1}/{a.folds} done")
    cv = pd.DataFrame(cv_rows)
    cv_mean = cv.groupby("model").mean(numeric_only=True).drop(columns="fold")
    cv_std = cv.groupby("model").std(numeric_only=True).drop(columns="fold")

    # ---- refit on full train, evaluate on holdout ----
    fmap = fit_road_freq(train)
    test_rows, fitted = [], {}
    for name, model in build_models(model_names).items():
        X_tr, X_te = make_X(train, name, fmap), make_X(test, name, fmap)
        X_te = align_categories(X_tr, X_te, name)
        t0 = time.time(); fit_model(model, name, X_tr, train[LABEL]); fit_s = time.time() - t0
        t0 = time.time(); p = model.predict_proba(X_te)[:, 1]; pred_ms = (time.time() - t0) * 1000
        test_rows.append({"model": name, **metrics(test[LABEL].values, p, test), "fit_s": fit_s, "predict_ms": pred_ms})
        fitted[name] = model
    test_df = pd.DataFrame(test_rows).set_index("model")

    lower_better = a.select_by in ("log_loss", "brier")
    ranked = cv_mean[a.select_by].sort_values(ascending=lower_better)
    # Selection rule: every model whose CV score is within one std of the leader
    # is statistically tied; among those pick the fastest at inference, since the
    # HMM scores every candidate of every GPS point.
    leader = ranked.index[0]
    tol = cv_std.loc[leader, a.select_by]
    gap = (ranked - ranked.iloc[0]).abs()
    tied = list(gap[gap <= tol].index)
    best = test_df.loc[tied, "predict_ms"].idxmin()

    # ---- save ----
    table = cv_mean.add_prefix("cv_").join(cv_std[[a.select_by, "f1", "top1_point_acc"]].add_prefix("cv_std_")).join(test_df.add_prefix("test_"))
    table = table.loc[ranked.index]
    table.to_csv(out / "model_comparison.csv")
    scorer = CandidateScorer(model=fitted[best], model_name=best, road_freq=fmap,
                             train_categories=None)
    joblib.dump(scorer, out / "best_model.joblib")

    show = ["accuracy", "f1", "roc_auc", "log_loss", "brier", "top1_point_acc"]
    fmt = lambda m, s: f"{m:.4f} ± {s:.4f}"
    lines = [f"# Candidate-segment model comparison\n",
             f"Data: `{a.data}` ({len(df)} rows, {df[g].nunique() if g else 'n/a'} groups by `{g}`), "
             f"positive rate {df[LABEL].mean():.3f}.\n",
             f"Split: grouped 80/20 holdout by `{g}`, {a.folds}-fold grouped CV on the 80%. "
             f"Selection metric: CV **{a.select_by}** ({'lower' if lower_better else 'higher'} is better), "
             "because the probabilities feed the HMM as emission scores and must be well calibrated.\n",
             f"## {a.folds}-fold CV (mean ± std)\n",
             "| Model | " + " | ".join(show) + " |", "|---" * (len(show) + 1) + "|"]
    for m in ranked.index:
        lines.append(f"| {m} | " + " | ".join(fmt(cv_mean.loc[m, c], cv_std.loc[m, c]) for c in show) + " |")
    lines += ["\n## Holdout test set\n",
              "| Model | " + " | ".join(show + ["fit_s", "predict_ms"]) + " |", "|---" * (len(show) + 3) + "|"]
    for m in ranked.index:
        lines.append(f"| {m} | " + " | ".join(f"{test_df.loc[m, c]:.4f}" for c in show) +
                     f" | {test_df.loc[m, 'fit_s']:.2f} | {test_df.loc[m, 'predict_ms']:.1f} |")
    lines += [f"\n## Selected model: **{best}**\n",
              f"Rule: models within one CV std ({tol:.4f}) of the best CV {a.select_by} "
              f"({leader}, {ranked.iloc[0]:.4f}) count as tied: {', '.join(tied)}. "
              f"Among those the fastest at inference wins: **{best}** "
              f"(CV {a.select_by} {ranked[best]:.4f}, {test_df.loc[best, 'predict_ms']:.1f} ms on the test set). "
              "Saved to `best_model.joblib`.\n",
              "Metric notes: `top1_point_acc` = share of GPS points where the highest-probability "
              "candidate is the true road (what the HMM effectively relies on). Accuracy/F1 use a 0.5 threshold.\n"]
    (out / "model_comparison.md").write_text("\n".join(lines))
    (out / "summary.json").write_text(json.dumps({"selected": best, "select_by": a.select_by, "tied": tied,
                                                  "ranking": ranked.round(5).to_dict()}, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
