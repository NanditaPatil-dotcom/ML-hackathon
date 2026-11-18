"""Raw GPS track cleaning.

Drops nulls/duplicates and filters points that imply physically impossible
speed or acceleration jumps relative to the previous point (urban vehicle
trajectory assumption: ~1 Hz GPS fix).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from geo import haversine_m

# Thresholds for vehicle GPS traces. The provided dataset includes German
# Autobahn driving with no general speed limit, so the cap must be well
# above typical urban speeds: 62.5 m/s ~= 225 km/h (comfortably above
# observed legitimate Autobahn cruising speeds, still rejects GPS
# teleports/jumps). 8 m/s^2 is above any real vehicle's sustained
# acceleration/braking.
MAX_SPEED_MPS = 62.5
MAX_ACCEL_MPS2 = 8.0
LON_RANGE = (-180.0, 180.0)
LAT_RANGE = (-90.0, 90.0)


@dataclass
class CleanStats:
    trip_id: str
    n_raw: int
    n_dropped_null: int = 0
    n_dropped_invalid_coord: int = 0
    n_dropped_duplicate: int = 0
    n_dropped_non_monotonic: int = 0
    n_dropped_speed_outlier: int = 0
    n_clean: int = 0

    def as_dict(self):
        return self.__dict__.copy()


def clean_track(trip_id: str, track: pd.DataFrame) -> tuple[pd.DataFrame, CleanStats]:
    stats = CleanStats(trip_id=trip_id, n_raw=len(track))
    df = track.copy()

    # 1. Nulls.
    before = len(df)
    df = df.dropna(subset=["lon", "lat", "timestamp"])
    stats.n_dropped_null = before - len(df)

    # 2. Implausible coordinates.
    before = len(df)
    df = df[
        df["lon"].between(*LON_RANGE) & df["lat"].between(*LAT_RANGE)
    ]
    stats.n_dropped_invalid_coord = before - len(df)

    # 3. Sort by timestamp, drop exact duplicate (lon, lat, timestamp) rows.
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    before = len(df)
    df = df.drop_duplicates(subset=["lon", "lat", "timestamp"]).reset_index(drop=True)
    stats.n_dropped_duplicate = before - len(df)

    # 4. Drop non-increasing timestamps (keep first occurrence of each ts
    #    progression; a GPS logger should not go backwards in time).
    before = len(df)
    keep = np.ones(len(df), dtype=bool)
    last_ts = -np.inf
    ts = df["timestamp"].to_numpy()
    for i in range(len(ts)):
        if ts[i] <= last_ts:
            keep[i] = False
        else:
            last_ts = ts[i]
    df = df[keep].reset_index(drop=True)
    stats.n_dropped_non_monotonic = before - len(df)

    # 5. Speed / acceleration outlier filtering, iterative: a point is
    #    dropped if the implied speed from the *last retained* point exceeds
    #    MAX_SPEED_MPS, or if the implied acceleration relative to the prior
    #    retained segment exceeds MAX_ACCEL_MPS2. This catches single-point
    #    GPS teleports/jumps without discarding the whole trip.
    if len(df) >= 2:
        keep_idx = [0]
        prev_speed = None
        for i in range(1, len(df)):
            j = keep_idx[-1]
            dt = df["timestamp"].iat[i] - df["timestamp"].iat[j]
            if dt <= 0:
                continue
            dist = haversine_m(
                df["lon"].iat[j], df["lat"].iat[j], df["lon"].iat[i], df["lat"].iat[i]
            )
            speed = dist / dt
            accel_ok = True
            if prev_speed is not None:
                accel_ok = abs(speed - prev_speed) / dt <= MAX_ACCEL_MPS2
            if speed <= MAX_SPEED_MPS and accel_ok:
                keep_idx.append(i)
                prev_speed = speed
            # else: drop point i, keep comparing future points against j
        n_before = len(df)
        df = df.iloc[keep_idx].reset_index(drop=True)
        stats.n_dropped_speed_outlier = n_before - len(df)

    stats.n_clean = len(df)
    return df, stats
