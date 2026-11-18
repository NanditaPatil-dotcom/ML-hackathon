"""Discover and load raw PathSnap trip files.

Verified layout (see Dataset Exploration doc + direct inspection of the
provided data): a partition directory (e.g. ``train/``, ``student_test/``)
contains one subdirectory per trip, named by a zero-padded trip id, e.g.::

    train/00000001/00000001.track
    train/00000001/00000001.nodes
    train/00000001/00000001.arcs
    train/00000001/00000001.route   (present for train / student_test; not
                                      for evaluator_test)

All four files are whitespace-separated with no header row.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class RawTrip:
    trip_id: str
    track: pd.DataFrame  # columns: lon, lat, timestamp
    nodes: np.ndarray  # shape (N, 2): lon, lat ; row index = node_id
    arcs: np.ndarray  # shape (M, 2) int: from_node_id, to_node_id ; row index = arc_id
    route: Optional[np.ndarray]  # 1D int array of arc_ids, or None


def discover_trips(partition_dir: str) -> list[str]:
    """Return sorted trip ids found as subdirectories of ``partition_dir``."""
    if not os.path.isdir(partition_dir):
        return []
    trip_ids = []
    for name in sorted(os.listdir(partition_dir)):
        sub = os.path.join(partition_dir, name)
        if os.path.isdir(sub) and os.path.isfile(os.path.join(sub, f"{name}.track")):
            trip_ids.append(name)
    return trip_ids


def load_trip(partition_dir: str, trip_id: str) -> RawTrip:
    base = os.path.join(partition_dir, trip_id, trip_id)

    track = pd.read_csv(
        f"{base}.track", sep=r"\s+", header=None, names=["lon", "lat", "timestamp"]
    )

    nodes = pd.read_csv(f"{base}.nodes", sep=r"\s+", header=None, names=["lon", "lat"]).to_numpy(
        dtype=float
    )

    arcs = pd.read_csv(
        f"{base}.arcs", sep=r"\s+", header=None, names=["from_node", "to_node"]
    ).to_numpy(dtype=np.int64)

    route_path = f"{base}.route"
    route = None
    if os.path.isfile(route_path) and os.path.getsize(route_path) > 0:
        route = pd.read_csv(route_path, sep=r"\s+", header=None, names=["arc_id"])[
            "arc_id"
        ].to_numpy(dtype=np.int64)

    return RawTrip(trip_id=trip_id, track=track, nodes=nodes, arcs=arcs, route=route)
