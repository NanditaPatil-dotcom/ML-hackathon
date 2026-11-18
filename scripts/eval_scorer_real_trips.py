"""Evaluate the selected scorer (CatBoost) + HMM/Viterbi on real trips.

Uses pathsnap.decoder.pipeline.reconstruct_trip_with_scorer, i.e. the same
preprocessing and features the model was trained on. Accuracy is over all
clean GPS points that have a ground-truth arc.

Usage:
    python scripts/eval_scorer_real_trips.py --trips-root "Student folder/student_test" \
        --model artifacts/models/catboost_best.joblib \
        --out results/metrics/scorer_hmm_real_trips_eval.csv
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pathsnap.decoder.pipeline import reconstruct_trip_with_scorer  # noqa: E402
from pathsnap.models.scorer import CandidateScorer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trips-root", required=True)
    ap.add_argument("--model", default="artifacts/models/catboost_best.joblib")
    ap.add_argument("--out", default="results/metrics/scorer_hmm_real_trips_eval.csv")
    args = ap.parse_args()

    scorer = CandidateScorer.load(args.model)
    cols = ["trip", "n_eval_points", "classifier_acc", "hmm_acc", "clf_continuity",
            "hmm_continuity", "seconds"]
    rows = []
    for tid in sorted(os.listdir(args.trips_root)):
        t0 = time.time()
        r = reconstruct_trip_with_scorer(args.trips_root, tid, scorer)
        if r is None or "hmm_acc" not in r["metrics"]:
            continue
        m = r["metrics"]
        rows.append([tid, m["n_eval_points"], round(m["clf_acc"], 1), round(m["hmm_acc"], 1),
                     round(m["clf_continuity"], 1), round(m["hmm_continuity"], 1),
                     round(time.time() - t0, 1)])
    n = sum(r[1] for r in rows)
    wavg = lambda i: round(sum(r[i] * r[1] for r in rows) / n, 1)
    rows.append(["ALL (point-weighted)", n, wavg(2), wavg(3), wavg(4), wavg(5),
                 round(sum(r[6] for r in rows), 1)])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
