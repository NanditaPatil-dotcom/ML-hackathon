"""Lightweight dataset-exploration script producing the EDA artefacts the
Deliverables doc asks for (trip counts, observation counts, missing-value /
anomaly summary, and a Folium map of one representative trip's raw GPS vs
ground-truth route vs road network).

Usage:
    python eda.py --data-root "/path/to/Student folder" --out-dir eda_outputs
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from io_utils import discover_trips, load_trip
from road_graph import RoadGraph


def summarize_partition(data_root: str, partition: str) -> pd.DataFrame:
    partition_dir = os.path.join(data_root, partition)
    trip_ids = discover_trips(partition_dir)
    rows = []
    for trip_id in trip_ids:
        raw = load_trip(partition_dir, trip_id)
        t = raw.track
        ts = t["timestamp"].to_numpy()
        dt = np.diff(ts) if len(ts) > 1 else np.array([])
        rows.append(
            {
                "partition": partition,
                "trip_id": trip_id,
                "n_points": len(t),
                "n_nulls": int(t.isna().sum().sum()),
                "n_duplicate_rows": int(t.duplicated().sum()),
                "duration_sec": float(ts.max() - ts.min()) if len(ts) else 0.0,
                "frac_dt_eq_1s": float(np.mean(dt == 1.0)) if len(dt) else float("nan"),
                "max_gap_sec": float(dt.max()) if len(dt) else float("nan"),
                "lon_min": t["lon"].min(),
                "lon_max": t["lon"].max(),
                "lat_min": t["lat"].min(),
                "lat_max": t["lat"].max(),
                "n_nodes": len(raw.nodes),
                "n_arcs": len(raw.arcs),
                "n_route_arcs": len(raw.route) if raw.route is not None else 0,
                "has_route": raw.route is not None,
            }
        )
    return pd.DataFrame(rows)


def make_folium_map(data_root: str, partition: str, trip_id: str, out_html: str) -> None:
    import folium

    partition_dir = os.path.join(data_root, partition)
    raw = load_trip(partition_dir, trip_id)
    t = raw.track
    center = [t["lat"].mean(), t["lon"].mean()]
    m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")

    # Raw GPS points.
    for _, r in t.iterrows():
        folium.CircleMarker(
            location=[r["lat"], r["lon"]], radius=2, color="blue", fill=True, fill_opacity=0.6
        ).add_to(m)

    # Ground-truth route polyline, if present.
    if raw.route is not None and len(raw.route) > 0:
        coords = []
        for arc_id in raw.route:
            from_node, to_node = raw.arcs[arc_id]
            coords.append([raw.nodes[from_node][1], raw.nodes[from_node][0]])
            coords.append([raw.nodes[to_node][1], raw.nodes[to_node][0]])
        folium.PolyLine(coords, color="green", weight=3, opacity=0.8, tooltip="Ground truth route").add_to(
            m
        )

    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    m.save(out_html)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", default="eda_outputs")
    ap.add_argument("--partitions", nargs="+", default=["train", "student_test"])
    ap.add_argument("--map-trip", default=None, help="trip_id to render a Folium map for")
    ap.add_argument("--map-partition", default="train")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_rows = []
    for partition in args.partitions:
        df = summarize_partition(args.data_root, partition)
        all_rows.append(df)
        print(f"[{partition}] {len(df)} trips summarized")
    summary = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    summary.to_csv(os.path.join(args.out_dir, "trip_inventory.csv"), index=False)
    print(summary.describe(include="all"))

    if args.map_trip:
        out_html = os.path.join(args.out_dir, f"{args.map_trip}_map.html")
        try:
            make_folium_map(args.data_root, args.map_partition, args.map_trip, out_html)
            print(f"wrote {out_html}")
        except ImportError:
            print("folium not installed; skip `pip install folium` to generate maps")


if __name__ == "__main__":
    main()
