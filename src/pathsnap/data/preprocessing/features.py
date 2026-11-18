"""Per-point candidate generation + feature engineering, and inter-point
transition features, for one cleaned trip.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .geo import bearing_deg, bearing_diff_deg, haversine_m
from .road_graph import RoadGraph

CANDIDATE_RADIUS_M = 40.0
CANDIDATE_K = 30


def build_candidates_table(
    trip_id: str,
    track: pd.DataFrame,
    graph: RoadGraph,
    gt_arc_per_point: Optional[np.ndarray] = None,
    radius_m: float = CANDIDATE_RADIUS_M,
    k: int = CANDIDATE_K,
) -> pd.DataFrame:
    """One row per (trip_id, point_idx, candidate_arc_id)."""
    lons = track["lon"].to_numpy()
    lats = track["lat"].to_numpy()
    ts = track["timestamp"].to_numpy()

    # Vehicle's instantaneous bearing at each point, from the previous point
    # (first point reuses the bearing to the next point).
    point_bearing = np.zeros(len(track))
    if len(track) >= 2:
        point_bearing[1:] = bearing_deg(lons[:-1], lats[:-1], lons[1:], lats[1:])
        point_bearing[0] = point_bearing[1] if len(track) > 1 else 0.0

    point_speed = np.zeros(len(track))
    if len(track) >= 2:
        dt = np.diff(ts)
        dt = np.where(dt <= 0, np.nan, dt)
        dist = haversine_m(lons[:-1], lats[:-1], lons[1:], lats[1:])
        point_speed[1:] = dist / dt

    rows = []
    for i in range(len(track)):
        cands = graph.query_candidates(lons[i], lats[i], radius_m=radius_m, k=k)
        gt_id = int(gt_arc_per_point[i]) if gt_arc_per_point is not None else None
        for c in cands:
            rows.append(
                {
                    "trip_id": trip_id,
                    "point_idx": i,
                    "timestamp": ts[i],
                    "lon": lons[i],
                    "lat": lats[i],
                    "candidate_arc_id": c.arc_id,
                    "dist_to_segment_m": c.dist_to_segment_m,
                    "arc_bearing_deg": c.arc_bearing_deg,
                    "point_bearing_deg": point_bearing[i],
                    "bearing_alignment_deg": float(
                        bearing_diff_deg(point_bearing[i], c.arc_bearing_deg)
                    ),
                    "arc_length_m": c.length_m,
                    "speed_limit_proxy_mps": c.speed_limit_proxy_mps,
                    "point_speed_mps": float(point_speed[i]),
                    "is_ground_truth_arc": int(gt_id is not None and c.arc_id == gt_id),
                }
            )
    return pd.DataFrame(rows)


def build_transitions_table(trip_id: str, track: pd.DataFrame) -> pd.DataFrame:
    """One row per consecutive GPS point pair: feeds HMM transition
    probabilities together with RoadGraph.arc_transitions() for road-network
    connectivity.
    """
    lons = track["lon"].to_numpy()
    lats = track["lat"].to_numpy()
    ts = track["timestamp"].to_numpy()
    if len(track) < 2:
        return pd.DataFrame(
            columns=["trip_id", "point_idx", "dt_sec", "point_transition_dist_m"]
        )
    dt = np.diff(ts)
    dist = haversine_m(lons[:-1], lats[:-1], lons[1:], lats[1:])
    return pd.DataFrame(
        {
            "trip_id": trip_id,
            "point_idx": np.arange(1, len(track)),
            "dt_sec": dt,
            "point_transition_dist_m": dist,
        }
    )
