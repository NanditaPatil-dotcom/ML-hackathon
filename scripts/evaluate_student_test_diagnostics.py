#!/usr/bin/env python3
"""Generate confusion matrices and trip-level exact-sequence diagnostics.

Uses the same deterministic 6,000-point training sample as
``evaluate_student_test.py`` and evaluates only the permitted student-test
partition. Sequence exact match requires every cleaned GPS observation in a
trip to have a candidate and the predicted road to match its reference road.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pathsnap.models.catboost_model  # noqa: E402,F401
import pathsnap.models.extra_trees  # noqa: E402,F401
import pathsnap.models.lightgbm_model  # noqa: E402,F401
import pathsnap.models.random_forest_model  # noqa: E402,F401
import pathsnap.models.rbf_svm  # noqa: E402,F401
from pathsnap.data.preprocessing.io_utils import load_trip  # noqa: E402
from pathsnap.evaluation.compare_models import load_dataset, subsample_by_point  # noqa: E402
from pathsnap.models.base import GROUP_COL, LABEL, MODEL_REGISTRY, NUMERIC_FEATURES, POINT_COL  # noqa: E402
from pathsnap.decoder.hmm_viterbi import HMMRouteDecoder  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default="results/preprocessing/candidate_data/train_candidates.parquet")
    ap.add_argument("--student-test", default="results/preprocessing/candidate_data/student_test_candidates.parquet")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--max-points", type=int, default=6000)
    ap.add_argument("--metrics-dir", default="results/metrics")
    ap.add_argument("--figures-dir", default="results/figures")
    args = ap.parse_args()

    train = subsample_by_point(load_dataset(args.train), args.max_points)
    test = load_dataset(args.student_test)
    expected = pd.read_csv(Path(args.student_test).parent / "student_test_trips.csv").set_index("trip_id")["n_clean"].to_dict()
    metrics_dir, figures_dir = Path(args.metrics_dir), Path(args.figures_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True); figures_dir.mkdir(parents=True, exist_ok=True)

    matrices, sequence_rows = {}, []
    figure, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.flat
    selected_name = "CatBoost"  # chosen by training-CV log loss in the preceding evaluation
    selected_model = None

    for axis, (name, spec) in zip(axes, MODEL_REGISTRY.items()):
        model = spec.build()
        spec.fit(model, spec.make_X(train), train[LABEL])
        probabilities = model.predict_proba(spec.make_X(test))[:, 1]
        labels = test[LABEL].to_numpy(dtype=int)
        predictions = (probabilities >= 0.5).astype(int)
        matrix = confusion_matrix(labels, predictions, labels=[0, 1])
        matrices[name] = {"labels": [0, 1], "matrix": matrix.tolist()}
        axis.imshow(matrix, cmap="Blues")
        axis.set(title=name, xlabel="Predicted", ylabel="Actual", xticks=[0, 1], yticks=[0, 1])
        for (row, col), count in np.ndenumerate(matrix):
            axis.text(col, row, str(count), ha="center", va="center", color="black")

        scored = test.assign(_probability=probabilities)
        point_rows = scored.loc[scored.groupby([GROUP_COL, POINT_COL])["_probability"].idxmax()]
        point_rows = point_rows.sort_values([GROUP_COL, POINT_COL])
        for trip_id, trip_points in point_rows.groupby(GROUP_COL, sort=True):
            n_expected = int(expected.get(trip_id, len(trip_points)))
            point_correct = trip_points[LABEL].astype(bool).to_numpy()
            sequence_rows.append({
                "model": name,
                "trip_id": trip_id,
                "cleaned_observations": n_expected,
                "observations_with_candidates": len(trip_points),
                "sequence_coverage": len(trip_points) / n_expected if n_expected else float("nan"),
                "pointwise_top1_accuracy": float(point_correct.mean()) if len(point_correct) else float("nan"),
                "exact_sequence_match": bool(len(trip_points) == n_expected and point_correct.all()),
            })
        if name == selected_name:
            selected_model = model

    for axis in list(axes)[len(MODEL_REGISTRY):]:
        axis.axis("off")
    figure.suptitle("Student-test confusion matrices (threshold = 0.5)")
    figure.tight_layout()
    figure.savefig(figures_dir / "student_test_confusion_matrices.png", dpi=180)
    plt.close(figure)
    (metrics_dir / "student_test_confusion_matrices.json").write_text(json.dumps(matrices, indent=2))

    seq_df = pd.DataFrame(sequence_rows)
    seq_df.to_csv(metrics_dir / "student_test_sequence_metrics_by_trip.csv", index=False)
    summary_rows = []
    for name, group in seq_df.groupby("model", sort=False):
        summary_rows.append({
            "model": name,
            "mean_trip_pointwise_top1_accuracy": group["pointwise_top1_accuracy"].mean(),
            "exact_sequence_match_rate": group["exact_sequence_match"].mean(),
            "mean_sequence_coverage": group["sequence_coverage"].mean(),
            "trips_with_exact_match": int(group["exact_sequence_match"].sum()),
            "student_test_trips": len(group),
        })

    # Add HMM exact-sequence score for the model selected on training CV.
    hmm_rows = []
    chosen = test.assign(_probability=selected_model.predict_proba(test[NUMERIC_FEATURES])[:, 1])
    for trip_id, trip_df in chosen.groupby(GROUP_COL, sort=True):
        raw = load_trip(str(Path(args.data_root) / "student_test"), str(trip_id))
        candidates, truths = {}, {}
        matched = trip_df[trip_df[LABEL] == 1].groupby(POINT_COL)["candidate_arc_id"].first()
        for point_idx, group in trip_df.groupby(POINT_COL, sort=True):
            p = group["_probability"].to_numpy(dtype=float)
            p = p / max(float(p.sum()), 1e-12)
            candidates[int(point_idx)] = [
                {"arc_id": int(a), "score": float(s)}
                for a, s in zip(group["candidate_arc_id"], p)
            ]
            truths[int(point_idx)] = int(matched.loc[point_idx]) if point_idx in matched.index else None
        decoded = HMMRouteDecoder(raw.arcs).decode(candidates)
        points = sorted(candidates)
        correct = [truths[p] is not None and truths[p] == pred for p, pred in zip(points, decoded)]
        n_expected = int(expected.get(trip_id, len(points)))
        hmm_rows.append({
            "model": f"{selected_name}+HMM",
            "trip_id": trip_id,
            "cleaned_observations": n_expected,
            "observations_with_candidates": len(points),
            "sequence_coverage": len(points) / n_expected if n_expected else float("nan"),
            "pointwise_top1_accuracy": float(np.mean(correct)) if correct else float("nan"),
            "exact_sequence_match": bool(len(points) == n_expected and all(correct)),
        })
    hmm_df = pd.DataFrame(hmm_rows)
    hmm_df.to_csv(metrics_dir / "student_test_hmm_sequence_metrics_by_trip.csv", index=False)
    all_seq = pd.concat([seq_df, hmm_df], ignore_index=True)
    final_rows = []
    for name, group in all_seq.groupby("model", sort=False):
        final_rows.append({
            "model": name,
            "mean_trip_pointwise_top1_accuracy": group["pointwise_top1_accuracy"].mean(),
            "exact_sequence_match_rate": group["exact_sequence_match"].mean(),
            "mean_sequence_coverage": group["sequence_coverage"].mean(),
            "trips_with_exact_match": int(group["exact_sequence_match"].sum()),
            "student_test_trips": len(group),
        })
    pd.DataFrame(final_rows).to_csv(metrics_dir / "student_test_sequence_metrics.csv", index=False)
    print(f"Confusion matrix figure: {figures_dir / 'student_test_confusion_matrices.png'}")
    print(f"Sequence metrics: {metrics_dir / 'student_test_sequence_metrics.csv'}")


if __name__ == "__main__":
    main()
