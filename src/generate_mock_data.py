"""Generate a mock candidate-segment dataset with the same columns the
preprocessing step will produce, so the training pipeline can be built and
tested before real data is ready.

Each GPS point gets 3-6 candidate road segments; exactly one is the true road
(label=1). The true candidate tends to be closer, better aligned in bearing and
have a shorter transition distance, with plenty of noise and overlap.
"""
import argparse

import numpy as np
import pandas as pd


def make_mock(n_points: int = 2000, n_roads: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    road_speed = rng.choice([30, 40, 50, 60, 80, 100], size=n_roads)
    rows = []
    for pid in range(n_points):
        trip_id = pid // 50
        k = rng.integers(3, 7)
        true_idx = rng.integers(k)
        roads = rng.choice(n_roads, size=k, replace=False)
        for j, road in enumerate(roads):
            is_true = j == true_idx
            if is_true:
                dist = abs(rng.normal(6, 6))
                bearing = abs(rng.normal(0, 20)) % 180
                trans = abs(rng.normal(25, 15))
            else:
                dist = abs(rng.normal(18, 12))
                bearing = abs(rng.normal(35, 45)) % 180
                trans = abs(rng.normal(55, 35))
            rows.append({
                "trip_id": trip_id,
                "point_id": pid,
                "distance_to_segment": round(dist, 2),
                "bearing_diff": round(bearing, 2),
                "speed_limit": int(road_speed[road]),
                "transition_distance": round(trans, 2),
                "candidate_road_id": int(road),
                "label": int(is_true),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/mock/candidates_mock.csv")
    ap.add_argument("--n-points", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    df = make_mock(a.n_points, seed=a.seed)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df)} rows ({df['point_id'].nunique()} points) to {a.out}")
