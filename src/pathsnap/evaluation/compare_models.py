"""Grouped CV + holdout comparison across candidate-scoring model families.

This is the core logic behind `scripts/train_all_models.py`: fit every
requested model on grouped k-fold CV (so candidates of one trip never
straddle train/validation), refit on the full training split, evaluate on a
held-out set, and pick a final model. Kept as a plain function (not a CLI)
so it's reusable from a script, a notebook, or a test.
"""
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold, train_test_split

from pathsnap.evaluation.metrics import compute_metrics
from pathsnap.models.base import LABEL, MODEL_REGISTRY, REQUIRED_COLUMNS, build_models, fit_road_freq
from pathsnap.models.scorer import CandidateScorer

SEED = 42


def load_dataset(path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    return df.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)


def _group_col(df: pd.DataFrame) -> str | None:
    for c in ("trip_id", "point_id"):
        if c in df.columns:
            return c
    return None


def _align_for(spec, X_tr, X_te):
    return X_te  # no categorical-dtype alignment needed for the current registry


def run_comparison(df: pd.DataFrame, model_names: list[str], folds: int = 5, select_by: str = "log_loss"):
    """Returns (report: dict, fitted_scorers: dict[name, CandidateScorer])."""
    g = _group_col(df)

    if g:
        tr_idx, te_idx = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(df, groups=df[g]))
    else:
        tr_idx, te_idx = train_test_split(np.arange(len(df)), test_size=0.2, stratify=df[LABEL], random_state=SEED)
    train, test = df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)

    splitter = GroupKFold(folds) if g else StratifiedKFold(folds, shuffle=True, random_state=SEED)
    split_args = (train, train[LABEL], train[g]) if g else (train, train[LABEL])
    cv_rows = []
    for fold, (i_tr, i_va) in enumerate(splitter.split(*split_args)):
        f_tr, f_va = train.iloc[i_tr], train.iloc[i_va]
        fmap = fit_road_freq(f_tr)
        for name, model in build_models(model_names).items():
            spec = MODEL_REGISTRY[name]
            X_tr, X_va = spec.make_X(f_tr, fmap), spec.make_X(f_va, fmap)
            spec.fit(model, X_tr, f_tr[LABEL])
            p = model.predict_proba(X_va)[:, 1]
            cv_rows.append({"model": name, "fold": fold, **compute_metrics(f_va[LABEL].values, p, f_va.reset_index(drop=True))})
    cv = pd.DataFrame(cv_rows)
    cv_mean = cv.groupby("model").mean(numeric_only=True).drop(columns="fold")
    cv_std = cv.groupby("model").std(numeric_only=True).drop(columns="fold")

    fmap = fit_road_freq(train)
    test_rows, scorers = [], {}
    for name, model in build_models(model_names).items():
        spec = MODEL_REGISTRY[name]
        X_tr, X_te = spec.make_X(train, fmap), spec.make_X(test, fmap)
        t0 = time.time(); spec.fit(model, X_tr, train[LABEL]); fit_s = time.time() - t0
        t0 = time.time(); p = model.predict_proba(X_te)[:, 1]; pred_ms = (time.time() - t0) * 1000
        test_rows.append({"model": name, **compute_metrics(test[LABEL].values, p, test), "fit_s": fit_s, "predict_ms": pred_ms})
        scorers[name] = CandidateScorer(model=model, model_name=name, road_freq=fmap,
                                        uses_native_categorical=spec.uses_native_categorical)
    test_df = pd.DataFrame(test_rows).set_index("model")

    lower_better = select_by in ("log_loss", "brier")
    ranked = cv_mean[select_by].sort_values(ascending=lower_better)
    # Models within one CV std of the leader are statistically tied; among
    # those, pick the fastest at inference (the HMM scores every candidate
    # of every GPS point, so that cost adds up).
    leader = ranked.index[0]
    tol = cv_std.loc[leader, select_by]
    gap = (ranked - ranked.iloc[0]).abs()
    tied = list(gap[gap <= tol].index)
    best = test_df.loc[tied, "predict_ms"].idxmin()

    report = {
        "group_col": g,
        "n_rows": len(df),
        "positive_rate": float(df[LABEL].mean()),
        "folds": folds,
        "select_by": select_by,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "test": test_df,
        "ranked": ranked,
        "leader": leader,
        "tol": tol,
        "tied": tied,
        "best": best,
    }
    return report, scorers


def render_markdown(report: dict, data_path: str) -> str:
    show = ["accuracy", "f1", "roc_auc", "log_loss", "brier", "top1_point_acc"]
    fmt = lambda m, s: f"{m:.4f} ± {s:.4f}"
    cv_mean, cv_std, test_df, ranked = report["cv_mean"], report["cv_std"], report["test"], report["ranked"]
    g, lower_better = report["group_col"], report["select_by"] in ("log_loss", "brier")
    lines = [
        "# Candidate-segment model comparison\n",
        f"Data: `{data_path}` ({report['n_rows']} rows, groups by `{g}`), positive rate {report['positive_rate']:.3f}.\n",
        f"Split: grouped 80/20 holdout by `{g}`, {report['folds']}-fold grouped CV on the 80%. "
        f"Selection metric: CV **{report['select_by']}** ({'lower' if lower_better else 'higher'} is better), "
        "because the probabilities feed the HMM as emission scores and must be well calibrated.\n",
        f"## {report['folds']}-fold CV (mean ± std)\n",
        "| Model | " + " | ".join(show) + " |", "|---" * (len(show) + 1) + "|",
    ]
    for m in ranked.index:
        lines.append(f"| {m} | " + " | ".join(fmt(cv_mean.loc[m, c], cv_std.loc[m, c]) for c in show) + " |")
    lines += ["\n## Holdout test set\n", "| Model | " + " | ".join(show + ["fit_s", "predict_ms"]) + " |", "|---" * (len(show) + 3) + "|"]
    for m in ranked.index:
        lines.append(f"| {m} | " + " | ".join(f"{test_df.loc[m, c]:.4f}" for c in show) +
                     f" | {test_df.loc[m, 'fit_s']:.2f} | {test_df.loc[m, 'predict_ms']:.1f} |")
    lines += [
        f"\n## Selected model: **{report['best']}**\n",
        f"Rule: models within one CV std ({report['tol']:.4f}) of the best CV {report['select_by']} "
        f"({report['leader']}, {ranked.iloc[0]:.4f}) count as tied: {', '.join(report['tied'])}. "
        f"Among those the fastest at inference wins: **{report['best']}** "
        f"(CV {report['select_by']} {ranked[report['best']]:.4f}, {test_df.loc[report['best'], 'predict_ms']:.1f} ms on the test set).\n",
        "Metric notes: `top1_point_acc` = share of GPS points where the highest-probability "
        "candidate is the true road (what the HMM effectively relies on). Accuracy/F1 use a 0.5 threshold.\n",
    ]
    return "\n".join(lines)
