"""Evaluate route reconstruction on real trips.

For each trip folder (<trip>/<trip>.track/.nodes/.arcs/.route) runs
reconstruct_route_for_trip and reports top-1 arc accuracy and continuity for:
  - classifier only (argmax per GPS point)
  - topology-only Viterbi (old transition model)
  - distance-aware Viterbi (Newson & Krumm transitions)

Usage:
    python scripts/eval_hmm_real_trips.py --trips-root "Student folder/student_test" \
        --model artifacts/best_model.joblib --out results/metrics/hmm_real_trips_eval.csv
"""
import argparse
import csv
import os
import sys
import time

import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pathsnap.decoder.hmm_viterbi as hv  # noqa: E402


def run(trips_root, model, topology_only=False):
    original = hv.HMMRouteDecoder.decode
    if topology_only:
        hv.HMMRouteDecoder.decode = lambda self, c, track=None: self.decode_topology(c)
    try:
        out = {}
        for tid in sorted(os.listdir(trips_root)):
            t0 = time.time()
            r = hv.reconstruct_route_for_trip(os.path.join(trips_root, tid), tid, model)
            if r is None:
                continue
            m = r["metrics"]
            m["n_points"] = len(r["clf_path"])
            m["seconds"] = round(time.time() - t0, 2)
            out[tid] = m
        return out
    finally:
        hv.HMMRouteDecoder.decode = original


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trips-root", required=True)
    ap.add_argument("--model", default="artifacts/best_model.joblib")
    ap.add_argument("--out", default="results/metrics/hmm_real_trips_eval.csv")
    args = ap.parse_args()

    model = joblib.load(args.model)
    new = run(args.trips_root, model)
    old = run(args.trips_root, model, topology_only=True)

    cols = ["trip", "n_points", "classifier_only_acc", "old_topology_hmm_acc",
            "new_distance_hmm_acc", "clf_continuity", "new_hmm_continuity", "seconds"]
    rows = []
    for tid, m in new.items():
        rows.append([tid, m["n_points"], round(m.get("clf_top1_acc", float("nan")), 1),
                     round(old[tid].get("hmm_top1_acc", float("nan")), 1),
                     round(m.get("hmm_top1_acc", float("nan")), 1),
                     round(m["clf_continuity"], 1), round(m["hmm_continuity"], 1), m["seconds"]])

    n = sum(r[1] for r in rows)
    def wavg(i):
        return round(sum(r[i] * r[1] for r in rows) / n, 1)
    rows.append(["ALL (point-weighted)", n, wavg(2), wavg(3), wavg(4), wavg(5), wavg(6),
                 round(sum(r[7] for r in rows), 2)])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
