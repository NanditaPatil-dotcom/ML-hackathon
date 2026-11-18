"""End-to-end PathSnap preprocessing + feature engineering CLI.

Usage:
    python pipeline.py --data-root "/path/to/Student folder" --out-dir outputs \
        [--partitions train student_test] [--radius-m 40] [--k 30] [--limit-trips N]

Produces, per partition, under --out-dir:
    <partition>_candidates.parquet
    <partition>_transitions.parquet
    <partition>_trips.parquet
And, once, for the training partition:
    train_folds.csv   (5-fold trip-level CV split)

See README.md for the full output schema.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

from clean import clean_track
from features import build_candidates_table, build_transitions_table
from io_utils import discover_trips, load_trip
from road_graph import RoadGraph
from route_match import match_route_to_track
from split import kfold_by_trip


def process_partition(
    data_root: str, partition: str, out_dir: str, radius_m: float, k: int, limit_trips: int | None
) -> None:
    partition_dir = os.path.join(data_root, partition)
    trip_ids = discover_trips(partition_dir)
    if not trip_ids:
        print(f"[{partition}] no trips found under {partition_dir}, skipping", file=sys.stderr)
        return
    if limit_trips:
        trip_ids = trip_ids[:limit_trips]

    all_candidates = []
    all_transitions = []
    trip_summaries = []

    for ti, trip_id in enumerate(trip_ids):
        t0 = time.time()
        raw = load_trip(partition_dir, trip_id)
        clean_df, stats = clean_track(trip_id, raw.track)

        graph = RoadGraph(raw.nodes, raw.arcs)

        gt_arc_per_point = None
        gt_match_dist = None
        if raw.route is not None and len(raw.route) > 0 and len(clean_df) > 0:
            gt_arc_per_point, gt_match_dist = match_route_to_track(
                raw.route, graph, clean_df["lon"].to_numpy(), clean_df["lat"].to_numpy()
            )

        cand_df = build_candidates_table(
            trip_id, clean_df, graph, gt_arc_per_point, radius_m=radius_m, k=k
        )
        trans_df = build_transitions_table(trip_id, clean_df)

        all_candidates.append(cand_df)
        all_transitions.append(trans_df)

        n_points = len(clean_df)
        miss_rate = float("nan")
        if gt_arc_per_point is not None and n_points > 0:
            matched_points = set(zip(cand_df["point_idx"], cand_df["candidate_arc_id"]))
            n_present = sum(
                1
                for p in range(n_points)
                if gt_arc_per_point[p] >= 0 and (p, int(gt_arc_per_point[p])) in matched_points
            )
            n_with_gt = int((gt_arc_per_point >= 0).sum())
            miss_rate = 1.0 - (n_present / n_with_gt) if n_with_gt else float("nan")

        trip_summaries.append(
            {
                "trip_id": trip_id,
                "n_nodes": len(raw.nodes),
                "n_arcs": len(raw.arcs),
                "n_route_arcs": len(raw.route) if raw.route is not None else 0,
                **stats.as_dict(),
                "candidate_miss_rate": miss_rate,
                "avg_match_dist_m": float(pd.Series(gt_match_dist).mean())
                if gt_match_dist is not None
                else float("nan"),
                "elapsed_sec": round(time.time() - t0, 2),
            }
        )
        print(
            f"[{partition}] ({ti + 1}/{len(trip_ids)}) {trip_id}: "
            f"{n_points} pts, {len(cand_df)} candidate rows, "
            f"miss_rate={miss_rate:.3f} ({time.time() - t0:.1f}s)",
            file=sys.stderr,
        )

    os.makedirs(out_dir, exist_ok=True)
    cand_all = pd.concat(all_candidates, ignore_index=True) if all_candidates else pd.DataFrame()
    trans_all = (
        pd.concat(all_transitions, ignore_index=True) if all_transitions else pd.DataFrame()
    )
    trips_all = pd.DataFrame(trip_summaries)

    cand_all.to_parquet(os.path.join(out_dir, f"{partition}_candidates.parquet"), index=False)
    trans_all.to_parquet(os.path.join(out_dir, f"{partition}_transitions.parquet"), index=False)
    trips_all.to_csv(os.path.join(out_dir, f"{partition}_trips.csv"), index=False)

    if partition == "train" and trip_ids:
        folds = kfold_by_trip(trip_ids)
        folds.to_csv(os.path.join(out_dir, "train_folds.csv"), index=False)

    print(f"[{partition}] done: {len(trip_ids)} trips -> {out_dir}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True, help="Path to the 'Student folder' directory")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--partitions", nargs="+", default=["train", "student_test"])
    ap.add_argument("--radius-m", type=float, default=40.0)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument(
        "--limit-trips",
        type=int,
        default=None,
        help="Process at most N trips per partition (useful for quick smoke tests)",
    )
    args = ap.parse_args()

    for partition in args.partitions:
        process_partition(
            args.data_root, partition, args.out_dir, args.radius_m, args.k, args.limit_trips
        )


if __name__ == "__main__":
    main()
