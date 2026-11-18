"""Lightweight dataset-exploration script producing the EDA artefacts the
Deliverables doc asks for (trip counts, observation counts, missing-value /
anomaly summary, and a Folium map of one representative trip's raw GPS vs
ground-truth route vs road network).

Usage:
    python eda.py --data-root "/path/to/Student folder" --out-dir eda_outputs
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .io_utils import discover_trips, load_trip
LON_RANGE = (-180.0, 180.0)
LAT_RANGE = (-90.0, 90.0)


def summarize_partition(data_root: str, partition: str) -> pd.DataFrame:
    partition_dir = os.path.join(data_root, partition)
    trip_ids = discover_trips(partition_dir)
    rows = []
    for trip_id in trip_ids:
        raw = load_trip(partition_dir, trip_id)
        t = raw.track
        numeric = t.apply(pd.to_numeric, errors="coerce")
        original_missing = t.isna()
        non_numeric = numeric.isna() & ~original_missing
        numeric_values = numeric.to_numpy(dtype=float)
        lon = numeric["lon"].to_numpy(dtype=float)
        lat = numeric["lat"].to_numpy(dtype=float)
        ts = numeric["timestamp"].to_numpy(dtype=float)
        missing_rows = original_missing.any(axis=1).to_numpy()
        non_numeric_rows = non_numeric.any(axis=1).to_numpy()
        nonfinite_rows = (~np.isfinite(numeric_values)).any(axis=1) & ~missing_rows & ~non_numeric_rows
        invalid_coord_rows = (
            ~np.isfinite(lon)
            | ~np.isfinite(lat)
            | (lon < LON_RANGE[0])
            | (lon > LON_RANGE[1])
            | (lat < LAT_RANGE[0])
            | (lat > LAT_RANGE[1])
        ) & ~missing_rows & ~non_numeric_rows
        valid_ts = ts[np.isfinite(ts)]
        dt = np.diff(ts) if len(ts) > 1 else np.array([])
        finite_dt = dt[np.isfinite(dt)]

        nodes = np.asarray(raw.nodes, dtype=float).reshape((-1, 2))
        arcs = np.asarray(raw.arcs, dtype=np.int64).reshape((-1, 2))
        node_valid = (
            np.isfinite(nodes).all(axis=1)
            & (nodes[:, 0] >= LON_RANGE[0])
            & (nodes[:, 0] <= LON_RANGE[1])
            & (nodes[:, 1] >= LAT_RANGE[0])
            & (nodes[:, 1] <= LAT_RANGE[1])
        )
        src, dst = arcs[:, 0], arcs[:, 1]
        endpoint_valid = (src >= 0) & (src < len(nodes)) & (dst >= 0) & (dst < len(nodes))
        valid_arc = endpoint_valid.copy()
        valid_arc[endpoint_valid] &= node_valid[src[endpoint_valid]] & node_valid[dst[endpoint_valid]]
        valid_node_ids = np.flatnonzero(node_valid)
        remap = np.full(len(nodes), -1, dtype=np.int64)
        remap[valid_node_ids] = np.arange(len(valid_node_ids))
        if len(valid_node_ids):
            graph = coo_matrix(
                (np.ones(int(valid_arc.sum()), dtype=np.uint8),
                 (remap[src[valid_arc]], remap[dst[valid_arc]])),
                shape=(len(valid_node_ids), len(valid_node_ids)),
            ).tocsr()
            n_components, component_labels = connected_components(
                graph, directed=False, return_labels=True
            )
            component_sizes = np.bincount(component_labels, minlength=n_components)
            largest_component_nodes = int(component_sizes.max()) if len(component_sizes) else 0
            isolated_nodes = int(np.count_nonzero(np.asarray(graph.getnnz(axis=0)).ravel() +
                                                   np.asarray(graph.getnnz(axis=1)).ravel() == 0))
        else:
            n_components = largest_component_nodes = isolated_nodes = 0
        valid_arc_count = int(valid_arc.sum())
        rows.append(
            {
                "partition": partition,
                "trip_id": trip_id,
                "n_points": len(t),
                "n_missing_rows": int(missing_rows.sum()),
                "n_non_numeric_rows": int(non_numeric_rows.sum()),
                "n_nonfinite_rows": int(nonfinite_rows.sum()),
                "n_invalid_coord_rows": int(invalid_coord_rows.sum()),
                "n_duplicate_rows": int(numeric.duplicated().sum()),
                "n_nonmonotonic_timestamp_steps": int(np.count_nonzero(dt <= 0)),
                "duration_sec": float(valid_ts.max() - valid_ts.min()) if len(valid_ts) else 0.0,
                "frac_dt_eq_1s": float(np.mean(finite_dt == 1.0)) if len(finite_dt) else float("nan"),
                "n_gaps_gt_2s": int(np.count_nonzero(finite_dt > 2.0)),
                "max_gap_sec": float(finite_dt.max()) if len(finite_dt) else float("nan"),
                "lon_min": float(np.nanmin(lon[np.isfinite(lon)])) if np.isfinite(lon).any() else float("nan"),
                "lon_max": float(np.nanmax(lon[np.isfinite(lon)])) if np.isfinite(lon).any() else float("nan"),
                "lat_min": float(np.nanmin(lat[np.isfinite(lat)])) if np.isfinite(lat).any() else float("nan"),
                "lat_max": float(np.nanmax(lat[np.isfinite(lat)])) if np.isfinite(lat).any() else float("nan"),
                "n_nodes": len(nodes),
                "n_invalid_nodes": int((~node_valid).sum()),
                "n_arcs": len(arcs),
                "n_invalid_arcs": int((~valid_arc).sum()),
                "n_duplicate_arcs": int(pd.DataFrame(arcs).duplicated().sum()),
                "n_self_loop_arcs": int(np.count_nonzero((src == dst) & valid_arc)),
                "n_valid_graph_nodes": len(valid_node_ids),
                "n_valid_graph_arcs": valid_arc_count,
                "n_weak_components": int(n_components),
                "n_isolated_nodes": isolated_nodes,
                "largest_component_nodes": largest_component_nodes,
                "largest_component_fraction": (
                    largest_component_nodes / len(valid_node_ids) if len(valid_node_ids) else float("nan")
                ),
                "n_route_arcs": len(raw.route) if raw.route is not None else 0,
                "has_route": raw.route is not None,
            }
        )
    return pd.DataFrame(rows)


def write_summary_outputs(summary: pd.DataFrame, out_dir: str) -> None:
    """Write per-trip inventory plus aggregated quality/connectivity evidence."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "trip_inventory.csv", index=False)
    aggregate_rows = []
    for partition, frame in summary.groupby("partition", sort=False):
        aggregate_rows.append({
            "partition": partition,
            "trips": int(len(frame)),
            "raw_gps_observations": int(frame["n_points"].sum()),
            "missing_gps_rows": int(frame["n_missing_rows"].sum()),
            "non_numeric_gps_rows": int(frame["n_non_numeric_rows"].sum()),
            "nonfinite_gps_rows": int(frame["n_nonfinite_rows"].sum()),
            "invalid_coordinate_rows": int(frame["n_invalid_coord_rows"].sum()),
            "duplicate_gps_rows": int(frame["n_duplicate_rows"].sum()),
            "nonmonotonic_timestamp_steps": int(frame["n_nonmonotonic_timestamp_steps"].sum()),
            "gaps_over_2s": int(frame["n_gaps_gt_2s"].sum()),
            "invalid_nodes": int(frame["n_invalid_nodes"].sum()),
            "invalid_arcs": int(frame["n_invalid_arcs"].sum()),
            "duplicate_arcs": int(frame["n_duplicate_arcs"].sum()),
            "self_loop_arcs": int(frame["n_self_loop_arcs"].sum()),
            "isolated_nodes": int(frame["n_isolated_nodes"].sum()),
            "trips_with_multiple_weak_components": int((frame["n_weak_components"] > 1).sum()),
            "weak_components_median_per_trip": float(frame["n_weak_components"].median()),
            "weak_components_max_per_trip": int(frame["n_weak_components"].max()),
            "largest_component_fraction_median": float(frame["largest_component_fraction"].median()),
            "largest_component_fraction_min": float(frame["largest_component_fraction"].min()),
        })
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(out / "eda_partition_summary.csv", index=False)
    (out / "eda_partition_summary.json").write_text(
        json.dumps(aggregate_rows, indent=2, allow_nan=False), encoding="utf-8"
    )

    lines = [
        "PATHSNAP DATA QUALITY AND ROAD-NETWORK CONNECTIVITY",
        "Scope: raw train and student_test partitions only; evaluator_test excluded.",
        "Connectivity uses weak components (directed arcs treated as undirected for component membership).",
        "",
    ]
    for row in aggregate_rows:
        lines.extend([
            f"[{row['partition']}] {row['trips']} trips; {row['raw_gps_observations']:,} raw GPS rows",
            "  GPS rows: "
            f"{row['missing_gps_rows']} missing, {row['non_numeric_gps_rows']} non-numeric, "
            f"{row['nonfinite_gps_rows']} non-finite, {row['invalid_coordinate_rows']} out-of-range coordinates, "
            f"{row['duplicate_gps_rows']} exact duplicates",
            f"  Timestamp sequence: {row['nonmonotonic_timestamp_steps']} non-increasing steps; "
            f"{row['gaps_over_2s']} gaps > 2 seconds",
            "  Road network: "
            f"{row['invalid_nodes']} invalid nodes, {row['invalid_arcs']} invalid arcs, "
            f"{row['duplicate_arcs']} duplicate arcs, {row['self_loop_arcs']} self-loops, "
            f"{row['isolated_nodes']} isolated nodes",
            "  Connectivity: "
            f"{row['trips_with_multiple_weak_components']} trips have multiple weak components; "
            f"median components/trip {row['weak_components_median_per_trip']:.1f}, "
            f"maximum {row['weak_components_max_per_trip']}; median largest-component node share "
            f"{row['largest_component_fraction_median']:.4f}, minimum "
            f"{row['largest_component_fraction_min']:.4f}",
            "",
        ])
    (out / "eda_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    make_quality_plots(aggregate, out)


def make_quality_plots(aggregate: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    partitions = aggregate["partition"].tolist()
    fields = [
        ("missing_gps_rows", "Missing GPS rows"),
        ("non_numeric_gps_rows", "Non-numeric GPS rows"),
        ("nonfinite_gps_rows", "Non-finite GPS rows"),
        ("invalid_coordinate_rows", "Out-of-range GPS coordinates"),
        ("duplicate_gps_rows", "Duplicate GPS rows"),
        ("nonmonotonic_timestamp_steps", "Non-increasing timestamp steps"),
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(fields))
    width = 0.36
    for i, partition in enumerate(partitions):
        values = [int(aggregate.loc[aggregate.partition == partition, key].iloc[0]) for key, _ in fields]
        bars = ax.bar(x + (i - (len(partitions) - 1) / 2) * width, values, width, label=partition)
        for bar, value in zip(bars, values):
            ax.annotate(str(value), (bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    max_count = max((int(aggregate[key].max()) for key, _ in fields), default=0)
    ax.set_ylim(0, max(1, max_count * 1.15))
    ax.set_xticks(x, [label for _, label in fields], rotation=22, ha="right")
    ax.set_ylabel("Count across partition")
    ax.set_title("Raw GPS data-quality checks")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "gps_data_quality.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for _, row in aggregate.iterrows():
        axes[0].bar(row["partition"], row["weak_components_median_per_trip"],
                    label=f"{row['partition']} (max {row['weak_components_max_per_trip']})")
        axes[1].bar(row["partition"], row["largest_component_fraction_median"])
    axes[0].set(title="Weak components per trip", ylabel="Median component count", yscale="log")
    axes[1].set(title="Largest component share", ylabel="Median fraction of valid nodes", ylim=(0, 1.05))
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Road-network connectivity (weak components)")
    fig.tight_layout()
    fig.savefig(out_dir / "road_network_connectivity.png", dpi=180)
    plt.close(fig)


def make_folium_map(data_root: str, partition: str, trip_id: str, out_html: str) -> None:
    import folium

    partition_dir = os.path.join(data_root, partition)
    raw = load_trip(partition_dir, trip_id)
    t = raw.track
    center = [t["lat"].mean(), t["lon"].mean()]
    m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")
    gps_layer = folium.FeatureGroup(name="Raw GPS observations", show=True).add_to(m)
    roads_layer = folium.FeatureGroup(name="Nearby directed road segments", show=True).add_to(m)
    route_layer = folium.FeatureGroup(name="Reference route", show=True).add_to(m)

    # Raw GPS points.
    for _, r in t.iterrows():
        folium.CircleMarker(
            location=[r["lat"], r["lon"]], radius=2, color="blue", fill=True, fill_opacity=0.6
        ).add_to(gps_layer)

    # Draw the local network around this trajectory, not the entire trip graph.
    nodes = np.asarray(raw.nodes, dtype=float)
    arcs = np.asarray(raw.arcs, dtype=np.int64)
    pad = 0.01
    lon_min, lon_max = float(t["lon"].min()) - pad, float(t["lon"].max()) + pad
    lat_min, lat_max = float(t["lat"].min()) - pad, float(t["lat"].max()) + pad
    in_bounds = (
        (nodes[:, 0] >= lon_min) & (nodes[:, 0] <= lon_max)
        & (nodes[:, 1] >= lat_min) & (nodes[:, 1] <= lat_max)
    )
    valid_arc = (
        (arcs[:, 0] >= 0) & (arcs[:, 0] < len(nodes))
        & (arcs[:, 1] >= 0) & (arcs[:, 1] < len(nodes))
    )
    valid_ids = np.flatnonzero(valid_arc)
    selected = valid_ids[in_bounds[arcs[valid_arc, 0]] & in_bounds[arcs[valid_arc, 1]]]
    if len(selected) > 3000:
        selected = selected[np.linspace(0, len(selected) - 1, 3000, dtype=int)]
    for arc_id in selected:
        u, v = arcs[arc_id]
        folium.PolyLine(
            [[nodes[u, 1], nodes[u, 0]], [nodes[v, 1], nodes[v, 0]]],
            color="#718096", weight=1, opacity=0.35,
        ).add_to(roads_layer)

    # Ground-truth route polyline, if present.
    if raw.route is not None and len(raw.route) > 0:
        coords = []
        for arc_id in raw.route:
            if arc_id < 0 or arc_id >= len(arcs):
                continue
            from_node, to_node = raw.arcs[arc_id]
            if from_node < 0 or to_node < 0 or from_node >= len(nodes) or to_node >= len(nodes):
                continue
            coords.append([raw.nodes[from_node][1], raw.nodes[from_node][0]])
            coords.append([raw.nodes[to_node][1], raw.nodes[to_node][0]])
        folium.PolyLine(coords, color="green", weight=3, opacity=0.8,
                        tooltip="Ground truth route").add_to(route_layer)

    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[lat_min, lon_min], [lat_max, lon_max]])

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
    write_summary_outputs(summary, args.out_dir)
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
