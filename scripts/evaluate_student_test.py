#!/usr/bin/env python3
"""Compare all registered candidate scorers on the permitted student-test set.

Training-side CV and model selection use only train trips. The ten student-test
trips are used once for development-stage comparison, as required by PathSnap.
The reserved evaluator-test partition is never read by this script.

Example:
  python scripts/evaluate_student_test.py \
    --train results/preprocessing/candidate_data/train_candidates.parquet \
    --student-test results/preprocessing/candidate_data/student_test_candidates.parquet \
    --data-root "/path/to/Student folder"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pathsnap.models.catboost_model  # noqa: E402,F401
import pathsnap.models.extra_trees  # noqa: E402,F401
import pathsnap.models.lightgbm_model  # noqa: E402,F401
import pathsnap.models.random_forest_model  # noqa: E402,F401
import pathsnap.models.rbf_svm  # noqa: E402,F401
from pathsnap.data.preprocessing.io_utils import load_trip  # noqa: E402
from pathsnap.evaluation.compare_models import load_dataset, subsample_by_point  # noqa: E402
from pathsnap.evaluation.metrics import compute_metrics  # noqa: E402
from pathsnap.models.base import GROUP_COL, LABEL, MODEL_REGISTRY, NUMERIC_FEATURES, POINT_COL  # noqa: E402
from pathsnap.decoder.hmm_viterbi import HMMRouteDecoder  # noqa: E402


def add_macro_metrics(row: dict, y_true, probabilities) -> dict:
    predictions = (probabilities >= 0.5).astype(int)
    row["macro_f1"] = float(f1_score(y_true, predictions, average="macro", zero_division=0))
    row["macro_precision"] = float(precision_score(y_true, predictions, average="macro", zero_division=0))
    row["macro_recall"] = float(recall_score(y_true, predictions, average="macro", zero_division=0))
    return row


def compute_route_continuity(predicted: list[int], point_ids: list[int], arcs: np.ndarray) -> float:
    valid, total = 0, 0
    for i in range(1, len(predicted)):
        if point_ids[i] != point_ids[i - 1] + 1:
            continue
        prev_arc, curr_arc = int(predicted[i - 1]), int(predicted[i])
        if not (0 <= prev_arc < len(arcs) and 0 <= curr_arc < len(arcs)):
            continue
        total += 1
        valid += int(prev_arc == curr_arc or int(arcs[prev_arc, 1]) == int(arcs[curr_arc, 0]))
    return float(valid / total) if total else float("nan")


def point_predictions(df: pd.DataFrame, probabilities: np.ndarray) -> tuple[list[int], list[int]]:
    scored = df[[GROUP_COL, POINT_COL, "candidate_arc_id"]].copy()
    scored["probability"] = probabilities
    chosen = scored.loc[scored.groupby([GROUP_COL, POINT_COL])["probability"].idxmax()]
    chosen = chosen.sort_values([GROUP_COL, POINT_COL])
    return chosen["candidate_arc_id"].astype(int).tolist(), chosen[POINT_COL].astype(int).tolist()


def evaluate_hmm(model, test_df: pd.DataFrame, data_root: Path) -> dict:
    """Compare classifier-only and simple Viterbi output for the selected model."""
    probabilities = model.predict_proba(test_df[NUMERIC_FEATURES])[:, 1]
    predictions, point_ids = point_predictions(test_df, probabilities)
    by_trip = test_df.assign(_probability=probabilities).groupby(GROUP_COL, sort=True)
    clf_correct = hmm_correct = total_points = 0
    clf_continuities, hmm_continuities = [], []

    for trip_id, trip_df in by_trip:
        raw = load_trip(str(data_root / "student_test"), str(trip_id))
        trip_df = trip_df.sort_values([POINT_COL, "candidate_arc_id"])
        candidates = {}
        truth = {}
        point_truth = trip_df[trip_df[LABEL] == 1].groupby(POINT_COL)["candidate_arc_id"].first()
        for point_idx, group in trip_df.groupby(POINT_COL, sort=True):
            scores = group["_probability"].to_numpy(dtype=float)
            scores = scores / max(float(scores.sum()), 1e-12)
            candidates[int(point_idx)] = [
                {"arc_id": int(arc), "score": float(score)}
                for arc, score in zip(group["candidate_arc_id"], scores)
            ]
            truth[int(point_idx)] = int(point_truth.loc[point_idx]) if point_idx in point_truth.index else None

        hmm_path = HMMRouteDecoder(raw.arcs).decode(candidates)
        ordered_points = sorted(candidates)
        clf_path = [max(candidates[p], key=lambda c: c["score"])["arc_id"] for p in ordered_points]
        total_points += len(ordered_points)
        clf_correct += sum(truth[p] is not None and pred == truth[p] for p, pred in zip(ordered_points, clf_path))
        hmm_correct += sum(truth[p] is not None and pred == truth[p] for p, pred in zip(ordered_points, hmm_path))
        clf_continuities.append(compute_route_continuity(clf_path, ordered_points, raw.arcs))
        hmm_continuities.append(compute_route_continuity(hmm_path, ordered_points, raw.arcs))

    return {
        "evaluated_observations_with_candidates": total_points,
        "classifier_top1_accuracy": clf_correct / total_points if total_points else None,
        "hmm_top1_accuracy": hmm_correct / total_points if total_points else None,
        "classifier_route_continuity": float(np.nanmean(clf_continuities)),
        "hmm_route_continuity": float(np.nanmean(hmm_continuities)),
    }


def make_plots(cv_df: pd.DataFrame, test_df: pd.DataFrame, hmm: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    order = test_df.sort_values("macro_f1", ascending=False)["model"].tolist()

    cv_plot = cv_df.set_index("model").loc[order]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(cv_plot.index, cv_plot["cv_macro_f1"], yerr=cv_plot["cv_macro_f1_std"], capsize=4, color="#4778b8")
    ax.set(title="Trip-grouped CV Macro F1", ylabel="Macro F1", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(out_dir / "cv_macro_f1.png", dpi=180)
    plt.close(fig)

    test_plot = test_df.set_index("model").loc[order]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(order)); width = 0.38
    ax.bar(x - width / 2, test_plot["top1_point_acc"], width, label="Top-1 accuracy", color="#26a69a")
    ax.bar(x + width / 2, test_plot["macro_f1"], width, label="Macro F1", color="#ef8a47")
    ax.set(title="Student-test candidate-scoring performance", ylabel="Score", xticks=x, xticklabels=order, ylim=(0, 1))
    ax.tick_params(axis="x", rotation=18); ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "student_test_top1_macro_f1.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(test_plot.index, test_plot["log_loss"], color="#8054a2")
    ax.set(title="Student-test Log Loss (lower is better)", ylabel="Log Loss")
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(out_dir / "student_test_log_loss.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    names = ["Classifier only", "Classifier + HMM/Viterbi"]
    x = np.arange(2); width = 0.36
    ax.bar(x - width / 2, [hmm["classifier_top1_accuracy"], hmm["hmm_top1_accuracy"]], width, label="Top-1 accuracy", color="#26a69a")
    ax.bar(x + width / 2, [hmm["classifier_route_continuity"], hmm["hmm_route_continuity"]], width, label="Route continuity", color="#8054a2")
    ax.set(title="Selected model: route-decoder impact", ylabel="Score", xticks=x, xticklabels=names, ylim=(0, 1))
    ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / "hmm_route_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default="results/preprocessing/candidate_data/train_candidates.parquet")
    ap.add_argument("--student-test", default="results/preprocessing/candidate_data/student_test_candidates.parquet")
    ap.add_argument("--data-root", required=True, help="Root containing train/ and student_test/ raw trip folders")
    ap.add_argument("--max-points", type=int, default=6000, help="Whole GPS points sampled from training data for tractable RBF SVM comparison")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--results-dir", default="results/metrics")
    ap.add_argument("--figures-dir", default="results/figures")
    args = ap.parse_args()

    train = load_dataset(args.train)
    student_test = load_dataset(args.student_test)
    train = subsample_by_point(train, args.max_points)
    if train[GROUP_COL].nunique() < args.folds:
        raise ValueError("Not enough distinct training trips for requested GroupKFold count.")
    groups = train[GROUP_COL].to_numpy()
    y = train[LABEL].to_numpy()
    results_dir, figures_dir = Path(args.results_dir), Path(args.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    cv_summaries, test_rows, fitted_models = [], [], {}
    splitter = GroupKFold(n_splits=args.folds)

    for model_name, spec in MODEL_REGISTRY.items():
        print(f"[{model_name}] {len(train):,} sampled train candidates; {len(student_test):,} student-test candidates", flush=True)
        fold_scores, fold_logloss = [], []
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(train, y, groups), start=1):
            model = spec.build()
            spec.fit(model, spec.make_X(train.iloc[train_idx]), y[train_idx])
            valid = train.iloc[valid_idx]
            p = model.predict_proba(spec.make_X(valid))[:, 1]
            metrics = compute_metrics(valid[LABEL].to_numpy(), p, valid)
            fold_scores.append(float(f1_score(valid[LABEL], (p >= 0.5).astype(int), average="macro", zero_division=0)))
            fold_logloss.append(float(metrics["log_loss"]))
            print(f"  fold {fold}/{args.folds}: macro_f1={fold_scores[-1]:.4f}, log_loss={fold_logloss[-1]:.4f}", flush=True)

        model = spec.build()
        t0 = time.perf_counter(); spec.fit(model, spec.make_X(train), train[LABEL]); fit_sec = time.perf_counter() - t0
        t0 = time.perf_counter(); p = model.predict_proba(spec.make_X(student_test))[:, 1]; predict_sec = time.perf_counter() - t0
        test_metrics = compute_metrics(student_test[LABEL].to_numpy(), p, student_test)
        test_metrics = add_macro_metrics(test_metrics, student_test[LABEL].to_numpy(), p)
        preds, point_ids = point_predictions(student_test, p)
        continuity = []
        scored_test = student_test.assign(_probability=p)
        for trip_id, trip_df in scored_test.groupby(GROUP_COL, sort=True):
            raw = load_trip(str(Path(args.data_root) / "student_test"), str(trip_id))
            trip_df = trip_df.sort_values([POINT_COL, "candidate_arc_id"])
            chosen = trip_df.loc[trip_df.groupby(POINT_COL)["_probability"].idxmax()].sort_values(POINT_COL)
            continuity.append(compute_route_continuity(chosen["candidate_arc_id"].astype(int).tolist(), chosen[POINT_COL].astype(int).tolist(), raw.arcs))

        cv_summaries.append({
            "model": model_name,
            "cv_macro_f1": float(np.mean(fold_scores)),
            "cv_macro_f1_std": float(np.std(fold_scores, ddof=1)),
            "cv_log_loss": float(np.mean(fold_logloss)),
            "cv_log_loss_std": float(np.std(fold_logloss, ddof=1)),
        })
        test_rows.append({
            "model": model_name,
            **test_metrics,
            "route_continuity": float(np.nanmean(continuity)),
            "train_seconds": fit_sec,
            "inference_seconds": predict_sec,
            "inference_ms_per_1000_candidates": predict_sec / len(student_test) * 1000_000,
        })
        # Persist progress in case the slower kernel model or decoder is interrupted.
        pd.DataFrame(cv_summaries).to_csv(results_dir / "student_test_model_cv.csv", index=False)
        pd.DataFrame(test_rows).to_csv(results_dir / "student_test_model_metrics.csv", index=False)
        fitted_models[model_name] = model
        print(f"  student-test: top1={test_metrics['top1_point_acc']:.4f}, macro_f1={test_metrics['macro_f1']:.4f}, log_loss={test_metrics['log_loss']:.4f}", flush=True)

    cv_df = pd.DataFrame(cv_summaries).sort_values("cv_log_loss").reset_index(drop=True)
    test_df = pd.DataFrame(test_rows)
    selected = str(cv_df.iloc[0]["model"])  # selection uses train CV only
    hmm_metrics = evaluate_hmm(fitted_models[selected], student_test, Path(args.data_root))
    cv_df.to_csv(results_dir / "student_test_model_cv.csv", index=False)
    test_df.to_csv(results_dir / "student_test_model_metrics.csv", index=False)
    (results_dir / "student_test_hmm_metrics.json").write_text(json.dumps({"selected_by_train_cv_log_loss": selected, **hmm_metrics}, indent=2))
    summary = {
        "train_rows_used": len(train), "train_trips_used": int(train[GROUP_COL].nunique()),
        "student_test_rows": len(student_test), "student_test_trips": int(student_test[GROUP_COL].nunique()),
        "selection_rule": "lowest mean trip-grouped CV log loss on training trips only",
        "selected_model": selected,
        "candidate_miss_rate_mean_over_student_test_trips": float(pd.read_csv(Path(args.train).parent / "student_test_trips.csv")["candidate_miss_rate"].mean()),
    }
    (results_dir / "student_test_evaluation_summary.json").write_text(json.dumps(summary, indent=2))
    make_plots(cv_df, test_df, hmm_metrics, figures_dir)
    print(f"Selected using training CV only: {selected}")
    print(f"Metrics: {results_dir / 'student_test_model_metrics.csv'}")
    print(f"Plots: {figures_dir}")


if __name__ == "__main__":
    main()
