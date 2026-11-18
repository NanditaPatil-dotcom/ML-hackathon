"""PathSnap Random Forest candidate-ranking baseline.

Expected dataset layout:
Student folder/
  train/<trip_id>/<trip_id>.track, .nodes, .arcs, .route
  student_test/<trip_id>/<trip_id>.track, .nodes, .arcs, .route

Important: .route contains a sequence of arc IDs, not one label per GPS row.
This script creates PROVISIONAL WEAK LABELS by assigning each GPS observation
its nearest reference-route arc in projected metres. It then trains a binary
candidate scorer. This is a practical baseline, not a substitute for validating
route-to-observation alignment and the official evaluation protocol.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from pyproj import Transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, classification_report, log_loss
import joblib

# Dataset is around 8.5E, 50N. EPSG:32632 is UTM zone 32N.
TO_METRES = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
FEATURES = [
    "candidate_distance_m", "midpoint_distance_m", "heading_difference_deg",
    "gps_speed_mps", "time_gap_s", "segment_length_m",
    "segment_bearing_deg", "candidate_rank", "gps_turn_angle_deg",
]


def read_trip(folder: Path):
    tid = folder.name
    track = np.loadtxt(folder / f"{tid}.track", dtype=float)
    nodes = np.loadtxt(folder / f"{tid}.nodes", dtype=float)
    arcs = np.loadtxt(folder / f"{tid}.arcs", dtype=int)
    route = np.loadtxt(folder / f"{tid}.route", dtype=int).reshape(-1)
    if track.ndim == 1: track = track.reshape(1, -1)
    if nodes.ndim == 1: nodes = nodes.reshape(1, -1)
    if arcs.ndim == 1: arcs = arcs.reshape(1, -1)
    # .track: longitude, latitude, timestamp; .nodes: longitude, latitude
    return tid, track, nodes, arcs, route


def project_xy(lonlat):
    x, y = TO_METRES.transform(lonlat[:, 0], lonlat[:, 1])
    return np.column_stack([x, y])


def bearing_deg(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_diff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def make_edges(nodes_xy, arcs):
    # arc_id is the zero-based row index in .arcs
    valid = (arcs[:, 0] >= 0) & (arcs[:, 0] < len(nodes_xy)) & (arcs[:, 1] >= 0) & (arcs[:, 1] < len(nodes_xy))
    valid_ids = np.flatnonzero(valid)
    edges, mids, lengths, bearings = [], [], [], []
    for aid in valid_ids:
        a, b = nodes_xy[arcs[aid, 0]], nodes_xy[arcs[aid, 1]]
        line = LineString([a, b])
        edges.append(line)
        mids.append(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
        lengths.append(line.length)
        bearings.append(bearing_deg(a, b))
    return valid_ids, edges, np.asarray(mids), np.asarray(lengths), np.asarray(bearings)


def process_trip(folder: Path, partition: str, k_candidates=20, search_radius_m=250.0):
    tid, track, nodes, arcs, route = read_trip(folder)
    gps_xy = project_xy(track[:, :2])
    nodes_xy = project_xy(nodes[:, :2])
    arc_ids, edges, mids, lengths, bearings = make_edges(nodes_xy, arcs)
    if len(edges) == 0 or len(route) == 0:
        return [], {"trip": tid, "observations": len(track), "candidate_misses": len(track), "used": 0}

    midpoint_tree = cKDTree(mids)
    # Build reference-route edge subset. Route entries are arc IDs.
    route_mask = np.isin(arc_ids, route)
    route_positions = np.flatnonzero(route_mask)
    if len(route_positions) == 0:
        warnings.warn(f"{tid}: no route IDs matched valid arc IDs")
        return [], {"trip": tid, "observations": len(track), "candidate_misses": len(track), "used": 0}
    route_mid_tree = cKDTree(mids[route_positions])

    # GPS movement features
    n = len(gps_xy)
    speeds = np.zeros(n)
    gaps = np.ones(n)
    headings = np.zeros(n)
    turns = np.zeros(n)
    for i in range(1, n):
        dt = max(float(track[i, 2] - track[i-1, 2]), 1e-3)
        gaps[i] = dt
        delta = gps_xy[i] - gps_xy[i-1]
        speeds[i] = np.linalg.norm(delta) / dt
        headings[i] = bearing_deg(gps_xy[i-1], gps_xy[i]) if np.linalg.norm(delta) > 0.01 else headings[i-1]
        if i > 1:
            turns[i] = angle_diff(headings[i], headings[i-1])

    rows = []
    candidate_misses = 0
    for i, pxy in enumerate(gps_xy):
        # Weak reference label: closest route arc to this GPS observation.
        rq = min(8, len(route_positions))
        _, rlocal = route_mid_tree.query(pxy, k=rq)
        rlocal = np.atleast_1d(rlocal)
        ref_best_pos = min(route_positions[rlocal], key=lambda pos: edges[pos].distance(Point(pxy)))
        ref_arc_id = int(arc_ids[ref_best_pos])

        q = min(k_candidates, len(edges))
        _, local = midpoint_tree.query(pxy, k=q)
        local = np.atleast_1d(local)
        cand_positions = []
        for pos in local:
            if edges[pos].distance(Point(pxy)) <= search_radius_m:
                cand_positions.append(int(pos))
        # Keep nearest candidates even if all exceed radius; record candidate miss below.
        if not cand_positions:
            cand_positions = [int(local[0])]
        cand_positions = sorted(set(cand_positions), key=lambda pos: edges[pos].distance(Point(pxy)))
        positive_found = any(int(arc_ids[pos]) == ref_arc_id for pos in cand_positions)
        if not positive_found:
            candidate_misses += 1

        for rank, pos in enumerate(cand_positions, start=1):
            line = edges[pos]
            dist = float(line.distance(Point(pxy)))
            seg_bearing = float(bearings[pos])
            hdiff = angle_diff(headings[i], seg_bearing) if speeds[i] > 0.5 else 0.0
            rows.append({
                "trip_id": tid,
                "observation_index": i,
                "candidate_arc_id": int(arc_ids[pos]),
                "candidate_distance_m": dist,
                "midpoint_distance_m": float(np.linalg.norm(mids[pos] - pxy)),
                "heading_difference_deg": hdiff,
                "gps_speed_mps": float(speeds[i]),
                "time_gap_s": float(gaps[i]),
                "segment_length_m": float(lengths[pos]),
                "segment_bearing_deg": seg_bearing,
                "candidate_rank": rank,
                "gps_turn_angle_deg": float(turns[i]),
                "label": int(int(arc_ids[pos]) == ref_arc_id),
                "reference_arc_id_weak": ref_arc_id,
                "partition": partition,
            })
    summary = {"trip": tid, "observations": n, "candidate_misses": candidate_misses,
               "candidate_miss_rate": candidate_misses / max(n, 1), "used": n - candidate_misses}
    return rows, summary


def find_trip_folders(root: Path, partition: str):
    base = root / partition
    return sorted(p for p in base.iterdir() if p.is_dir() and (p / f"{p.name}.track").exists())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to extracted Student folder")
    parser.add_argument("--k", type=int, default=20, help="Road candidates per GPS point")
    parser.add_argument("--radius", type=float, default=250.0, help="Candidate search radius in metres")
    parser.add_argument("--trees", type=int, default=400)
    parser.add_argument("--output", default="rf_results")
    args = parser.parse_args()

    root = Path(args.data)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    train_rows, train_summaries = [], []
    for folder in find_trip_folders(root, "train"):
        print(f"Processing training trip {folder.name}...")
        rows, summary = process_trip(folder, "train", args.k, args.radius)
        train_rows.extend(rows); train_summaries.append(summary)
    if not train_rows:
        raise RuntimeError("No training candidates generated. Check --data path and dataset structure.")

    train_df = pd.DataFrame(train_rows)
    # Remove GPS observations whose candidate set did not contain the weak positive.
    usable = train_df.groupby("observation_index", sort=False) if False else None
    obs_has_positive = train_df.groupby(["trip_id", "observation_index"])["label"].transform("max") == 1
    train_df = train_df.loc[obs_has_positive].copy()
    if train_df["trip_id"].nunique() < 2:
        raise RuntimeError("Need at least two usable training trips for trip-aware validation.")

    X = train_df[FEATURES]
    y = train_df["label"].astype(int)
    groups = train_df["trip_id"]
    n_splits = min(5, groups.nunique())
    cv = GroupKFold(n_splits=n_splits)
    model = RandomForestClassifier(
        n_estimators=args.trees,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    # Out-of-fold probabilities ensure trips are held out as groups.
    oof_prob = cross_val_predict(model, X, y, groups=groups, cv=cv,
                                 method="predict_proba", n_jobs=1)[:, 1]
    train_df["oof_positive_probability"] = oof_prob
    pred = (oof_prob >= 0.5).astype(int)
    print("\nTrip-grouped candidate-level CV:")
    print(classification_report(y, pred, zero_division=0))
    print("Candidate-level log loss:", round(log_loss(y, np.column_stack([1-oof_prob, oof_prob]), labels=[0, 1]), 5))

    # Observation-level top-1: choose candidate with highest OOF probability.
    train_df["rank_score"] = oof_prob
    best = train_df.loc[train_df.groupby(["trip_id", "observation_index"])["rank_score"].idxmax()]
    top1 = (best["candidate_arc_id"].to_numpy() == best["reference_arc_id_weak"].to_numpy()).mean()
    print("Weak-label observation Top-1 accuracy:", round(float(top1), 5))

    # Fit final model on all 80-trip training candidate rows.
    model.fit(X, y)
    joblib.dump({"model": model, "features": FEATURES, "k_candidates": args.k,
                 "search_radius_m": args.radius, "crs": "EPSG:32632"}, out / "random_forest_pathsnap.joblib")
    train_df.to_csv(out / "training_candidate_results.csv", index=False)
    pd.DataFrame(train_summaries).to_csv(out / "training_candidate_miss_summary.csv", index=False)

    # Student-test evaluation is development-only. Labels are weakly derived from that partition's .route.
    test_rows, test_summaries = [], []
    for folder in find_trip_folders(root, "student_test"):
        print(f"Processing student-test trip {folder.name}...")
        rows, summary = process_trip(folder, "student_test", args.k, args.radius)
        test_rows.extend(rows); test_summaries.append(summary)
    if test_rows:
        test_df = pd.DataFrame(test_rows)
        test_df["positive_probability"] = model.predict_proba(test_df[FEATURES])[:, 1]
        test_df["rank_score"] = test_df["positive_probability"]
        test_obs_has_positive = test_df.groupby(["trip_id", "observation_index"])["label"].transform("max") == 1
        evaluable = test_df.loc[test_obs_has_positive].copy()
        best_test = evaluable.loc[evaluable.groupby(["trip_id", "observation_index"])["rank_score"].idxmax()]
        test_top1 = (best_test["candidate_arc_id"] == best_test["reference_arc_id_weak"]).mean() if len(best_test) else float("nan")
        print("Student-test weak-label Top-1 accuracy:", round(float(test_top1), 5) if np.isfinite(test_top1) else "N/A")
        test_df.to_csv(out / "student_test_candidate_results.csv", index=False)
        pd.DataFrame(test_summaries).to_csv(out / "student_test_candidate_miss_summary.csv", index=False)

    print(f"\nSaved model and CSV outputs in: {out.resolve()}")
    print("WARNING: Report these as weak-label development metrics, not official evaluator accuracy.")

if __name__ == "__main__":
    main()
