#!/usr/bin/env python3
"""Bayesian hyperparameter optimisation (Optuna TPE) for the selected model, CatBoost.

Search: trip-grouped 3-fold CV on a point-level subsample of the training
trips (whole GPS points kept together), objective = mean CV log loss
(minimised; the HMM consumes the probabilities, so calibration matters more
than a 0.5-threshold metric). The best parameters are then compared with the
untuned defaults on the SAME data and metrics: 5-fold trip-grouped CV on all
training trips (folds from train_folds.csv) and the 10 student_test trips.
Evaluator-test data is never read.

    python scripts/tune_catboost.py \
        --train data_processing_outputs/train_candidates.parquet \
        --folds data_processing_outputs/train_folds.csv \
        --test  data_processing_outputs/student_test_candidates.parquet \
        --trials 25

Outputs:
    results/bayesian/catboost_trials.csv, catboost_best_params.json, catboost_history.png
    results/metrics/catboost_tuned_vs_untuned.csv
    artifacts/models/catboost_tuned.joblib   (CandidateScorer; not wired in automatically)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import yaml
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.evaluation.compare_models import SEED, load_dataset, subsample_by_point  # noqa: E402
from pathsnap.evaluation.metrics import compute_metrics  # noqa: E402
from pathsnap.models.base import GROUP_COL, LABEL, NUMERIC_FEATURES  # noqa: E402
from pathsnap.models.catboost_model import load_params  # noqa: E402
from pathsnap.models.scorer import CandidateScorer  # noqa: E402


def fit(params, df):
    m = CatBoostClassifier(**{**params, "verbose": 0, "random_seed": SEED, "thread_count": -1})
    m.fit(df[NUMERIC_FEATURES], df[LABEL])
    return m


def macro_f1(y, p):
    return f1_score(y, (p >= 0.5).astype(int), average="macro", zero_division=0)


def evaluate_cv(params, df, fold_of_trip):
    """5-fold trip-grouped CV with the folds from train_folds.csv."""
    folds = df[GROUP_COL].map(fold_of_trip)
    rows = []
    for f in sorted(folds.unique()):
        tr, va = df[folds != f], df[folds == f]
        m = fit(params, tr)
        p = m.predict_proba(va[NUMERIC_FEATURES])[:, 1]
        r = compute_metrics(va[LABEL].to_numpy(), p, va)
        r["macro_f1"] = macro_f1(va[LABEL].to_numpy(), p)
        rows.append(r)
    return pd.DataFrame(rows).mean().to_dict()


def evaluate_test(params, train, test):
    t0 = time.time()
    m = fit(params, train)
    fit_s = time.time() - t0
    t0 = time.time()
    p = m.predict_proba(test[NUMERIC_FEATURES])[:, 1]
    inf_s = time.time() - t0
    r = compute_metrics(test[LABEL].to_numpy(), p, test)
    r["macro_f1"] = macro_f1(test[LABEL].to_numpy(), p)
    r["fit_s"], r["inference_ms_per_1k"] = fit_s, 1000 * inf_s / (len(test) / 1000)
    return r, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--search-points", type=int, default=60000)
    ap.add_argument("--search-folds", type=int, default=3)
    a = ap.parse_args()

    train, test = load_dataset(a.train), load_dataset(a.test)
    folds = pd.read_csv(a.folds, dtype={"trip_id": str})
    fold_of_trip = dict(zip(folds["trip_id"], folds["fold"]))
    train[GROUP_COL] = train[GROUP_COL].astype(str)
    sub = subsample_by_point(train, a.search_points)
    print(f"search data: {len(sub):,} rows, {sub[GROUP_COL].nunique()} trips", flush=True)
    gkf = list(GroupKFold(n_splits=a.search_folds).split(sub, groups=sub[GROUP_COL]))

    def objective(trial):
        params = dict(
            iterations=trial.suggest_int("iterations", 200, 1000, step=100),
            depth=trial.suggest_int("depth", 4, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 3.0),
            random_strength=trial.suggest_float("random_strength", 0.1, 5.0, log=True),
            border_count=trial.suggest_categorical("border_count", [32, 64, 128, 254]),
            loss_function="Logloss",
        )
        losses = []
        for tr, va in gkf:
            m = fit(params, sub.iloc[tr])
            p = m.predict_proba(sub.iloc[va][NUMERIC_FEATURES])[:, 1]
            losses.append(log_loss(sub.iloc[va][LABEL], np.clip(p, 1e-7, 1 - 1e-7)))
        return float(np.mean(losses))

    # The untuned defaults are trial 0, so the search can only improve on them.
    default = {k: v for k, v in load_params(str(ROOT / "configs" / "models.yaml")).items()
               if k not in ("verbose", "random_seed")}
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.enqueue_trial({"iterations": 500, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3.0,
                         "bagging_temperature": 1.0, "random_strength": 1.0, "border_count": 254})
    t0 = time.time()
    study.optimize(objective, n_trials=a.trials,
                   callbacks=[lambda s, t: print(f"trial {t.number}: {t.value:.4f} {t.params} "
                                                 f"({time.time() - t0:.0f}s)", flush=True)])
    best = {**study.best_params, "loss_function": "Logloss"}
    search_s = time.time() - t0

    out_b, out_m = ROOT / "results" / "bayesian", ROOT / "results" / "metrics"
    out_b.mkdir(parents=True, exist_ok=True)
    out_m.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(out_b / "catboost_trials.csv", index=False)
    (out_b / "catboost_best_params.json").write_text(json.dumps(
        {"best_params": best, "search_cv_log_loss": study.best_value, "n_trials": a.trials,
         "search_rows": len(sub), "search_folds": a.search_folds, "search_seconds": round(search_s)},
        indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vals = [t.value for t in study.trials]
    plt.figure(figsize=(7, 4))
    plt.plot(vals, "o-", label="trial CV log loss")
    plt.plot(np.minimum.accumulate(vals), "r-", label="best so far")
    plt.xlabel("trial"); plt.ylabel("3-fold trip-grouped CV log loss"); plt.legend()
    plt.title("CatBoost Bayesian optimisation (Optuna TPE)"); plt.tight_layout()
    plt.savefig(out_b / "catboost_history.png", dpi=150)

    print("comparing untuned vs tuned on full train CV and student_test ...", flush=True)
    rows, models = [], {}
    for name, params in [("untuned", {**default, "loss_function": "Logloss"}), ("tuned", best)]:
        cv = evaluate_cv(params, train, fold_of_trip)
        te, m = evaluate_test(params, train, test)
        models[name] = m
        rows.append({"model": name, **{f"cv_{k}": v for k, v in cv.items()},
                     **{f"test_{k}": v for k, v in te.items()}, "params": json.dumps(params)})
    pd.DataFrame(rows).to_csv(out_m / "catboost_tuned_vs_untuned.csv", index=False)
    art = ROOT / "artifacts" / "models"
    art.mkdir(parents=True, exist_ok=True)
    joblib.dump(CandidateScorer(model=models["tuned"], model_name="CatBoost (tuned)"),
                art / "catboost_tuned.joblib")
    cols = ["model", "cv_macro_f1", "cv_log_loss", "test_top1_point_acc", "test_macro_f1",
            "test_log_loss", "test_inference_ms_per_1k"]
    print(pd.DataFrame(rows)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
