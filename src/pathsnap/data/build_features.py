"""Candidate-rank feature, and a mock feature generator matching the real
preprocessing schema (data_processing/pipeline.py's candidates parquet).
"""
import numpy as np
import pandas as pd

from pathsnap.models.base import GROUP_COL, POINT_COL


def add_candidate_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Rank candidates within each (trip_id, point_idx) group by
    dist_to_segment_m ascending (0 = nearest). Trip-independent substitute
    for candidate_arc_id, which isn't stable across trips. Must be computed
    over the full group, so call this once on a whole dataset/batch before
    scoring — not derivable from a single candidate row."""
    out = df.copy()
    out["candidate_rank"] = (
        out.groupby([GROUP_COL, POINT_COL])["dist_to_segment_m"].rank(method="first").astype(int) - 1
    )
    return out


def make_mock_features(n_points: int = 2000, n_roads: int = 300, seed: int = 42) -> pd.DataFrame:
    """Stands in for data_processing/pipeline.py until/when you want to
    smoke-test the shape of the pipeline without the real dataset. Emits the
    same columns the real candidates parquet has (trip_id, point_idx,
    dist_to_segment_m, bearing_alignment_deg, speed_limit_proxy_mps,
    point_speed_mps, arc_length_m, candidate_rank, is_ground_truth_arc).
    Each GPS point gets 3-6 candidates; exactly one is the true road."""
    rng = np.random.default_rng(seed)
    arc_length = rng.choice([15, 30, 60, 120, 300, 1000], size=n_roads).astype(float)
    rows = []
    for pid in range(n_points):
        trip_id = f"mock_{pid // 50:04d}"
        k = rng.integers(3, 7)
        true_idx = rng.integers(k)
        roads = rng.choice(n_roads, size=k, replace=False)
        for j, road in enumerate(roads):
            is_true = j == true_idx
            if is_true:
                dist, bearing, speed = abs(rng.normal(6, 6)), abs(rng.normal(0, 20)) % 180, abs(rng.normal(15, 5))
            else:
                dist, bearing, speed = abs(rng.normal(18, 12)), abs(rng.normal(35, 45)) % 180, abs(rng.normal(15, 5))
            rows.append({
                "trip_id": trip_id,
                "point_idx": pid % 50,
                "dist_to_segment_m": round(dist, 2),
                "bearing_alignment_deg": round(bearing, 2),
                "speed_limit_proxy_mps": round(float(rng.choice([8, 11, 14, 17, 22, 28])), 2),
                "point_speed_mps": round(speed, 2),
                "arc_length_m": round(float(arc_length[road]), 2),
                "is_ground_truth_arc": int(is_true),
            })
    df = pd.DataFrame(rows)
    return add_candidate_rank(df)
