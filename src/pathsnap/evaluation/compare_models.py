"""Grouped CV + holdout comparison across candidate-scoring model families.

This is the core logic behind `scripts/train_all_models.py`: fit every
requested model on trip-grouped k-fold CV (so candidates of one trip never
straddle train/validation), refit on the full training split, evaluate on a
held-out set, and pick a final model. Kept as a plain function (not a CLI)
so it's reusable from a script, a notebook, or a test.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from pathsnap.data.build_features import add_candidate_rank
from pathsnap.evaluation.metrics import compute_metrics
from pathsnap.models.base import GROUP_COL, LABEL, MODEL_REGISTRY, POINT_COL, REQUIRED_COLUMNS, build_models
from pathsnap.models.scorer import CandidateScorer

SEED = 42


def load_dataset(path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "candidate_rank" not in df.columns:
        df = add_candidate_rank(df)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    return df.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)


def subsample_by_point(df: pd.DataFrame, max_points: int, seed: int = SEED) -> pd.DataFrame:
    """Keep only `max_points` whole (trip_id, point_idx) groups (every
    candidate of a kept point stays together). Use this to keep CV
    tractable for models that don't scale to the full real dataset (RBF SVM
    is roughly O(n^2)-O(n^3) in rows); CatBoost can still be trained
    separately on the full data since gradient boosting scales linearly."""
    if GROUP_COL not in df.columns or POINT_COL not in df.columns:
        return df
    points = df[[GROUP_COL, POINT_COL]].drop_duplicates()
    if len(points) <= max_points:
        return df
    keep = points.sample(max_points, random_state=seed)
    return df.merge(keep, on=[GROUP_COL, POINT_COL], how="inner").reset_index(drop=True)


def run_comparison(df: pd.DataFrame, model_names: list[str], folds: int = 5, select_by: str = "log_loss"):
    """Returns (report: dict, fitted_scorers: dict[name, CandidateScorer])."""
    g = GROUP_COL if GROUP_COL in df.columns else None

    if g:
        tr_idx, te_idx = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(df, groups=df[g]))
    else:
        from sklearn.model_selection import train_test_split
        tr_idx, te_idx = train_test_split(np.arange(len(df)), test_size=0.2, stratify=df[LABEL], random_state=SEED)
    train, test = df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)

    splitter = GroupKFold(folds)
    cv_rows = []
    for fold, (i_tr, i_va) in enumerate(splitter.split(train, train[LABEL], train[g])):
        f_tr, f_va = train.iloc[i_tr], train.iloc[i_va]
        for name, model in build_models(model_names).items():
            spec = MODEL_REGISTRY[name]
            X_tr, X_va = spec.make_X(f_tr), spec.make_X(f_va)
            spec.fit(model, X_tr, f_tr[LABEL])
            p = model.predict_proba(X_va)[:, 1]
            cv_rows.append({"model": name, "fold": fold, **compute_metrics(f_va[LABEL].values, p, f_va.reset_index(drop=True))})
    cv = pd.DataFrame(cv_rows)
    cv_mean = cv.groupby("model").mean(numeric_only=True).drop(columns="fold")
    cv_std = cv.groupby("model").std(numeric_only=True).drop(columns="fold")

    test_rows, scorers = [], {}
    for name, model in build_models(model_names).items():
        spec = MODEL_REGISTRY[name]
        X_tr, X_te = spec.make_X(train), spec.make_X(test)
        t0 = time.time(); spec.fit(model, X_tr, train[LABEL]); fit_s = time.time() - t0
        t0 = time.time(); p = model.predict_proba(X_te)[:, 1]; pred_ms = (time.time() - t0) * 1000
        test_rows.append({"model": name, **compute_metrics(test[LABEL].values, p, test), "fit_s": fit_s, "predict_ms": pred_ms,
                          "predict_ms_per_1k": pred_ms / max(len(test), 1) * 1000})
        scorers[name] = CandidateScorer(model=model, model_name=name)
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


def leaderboard(report: dict) -> pd.DataFrame:
    """One ranked row per model: CV log-loss / accuracy / top-1, holdout
    log-loss / accuracy, and inference time. Ordered by the selection metric."""
    cv, test = report["cv_mean"], report["test"]
    order = list(report["ranked"].index)
    lb = pd.DataFrame({
        "cv_log_loss": cv["log_loss"], "cv_accuracy": cv["accuracy"], "cv_top1_point_acc": cv["top1_point_acc"],
        "test_log_loss": test["log_loss"], "test_accuracy": test["accuracy"],
        "fit_s": test["fit_s"], "inference_ms_per_1k": test["predict_ms_per_1k"],
    }).loc[order]
    lb.insert(0, "rank", range(1, len(lb) + 1))
    lb["selected"] = [m == report["best"] for m in lb.index]
    lb.index.name = "model"
    return lb


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
        "## Ranking\n",
        "| # | Model | CV log_loss | CV accuracy | CV top1_point_acc | Test log_loss | Test accuracy | Fit (s) | Inference (ms / 1k candidates) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m, r in leaderboard(report).iterrows():
        name = f"**{m}** (selected)" if r["selected"] else m
        lines.append(f"| {r['rank']} | {name} | {r['cv_log_loss']:.4f} | {r['cv_accuracy']:.4f} | "
                     f"{r['cv_top1_point_acc']:.4f} | {r['test_log_loss']:.4f} | {r['test_accuracy']:.4f} | "
                     f"{r['fit_s']:.2f} | {r['inference_ms_per_1k']:.2f} |")
    lines += [
        "",
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
