"""Align the chronological ground-truth arc sequence (.route) to individual
GPS points, so the candidate table can carry an ``is_ground_truth_arc``
label for supervised training.

The .route file is a path through the graph (a sequence of arc ids), not a
per-point label -- there is no guaranteed 1:1 correspondence between GPS
rows and route rows (route may have ~500 arcs, track may have ~1300 points,
and a single arc can be traversed over several GPS fixes, or a short arc
skipped between two fixes). We instead snap each GPS point to the single
closest point on the route's own polyline, using a sliding window around
the previous match's route position so the mapping is kept monotonic
(progress along the route should not jump backwards for a single trip).
"""
from __future__ import annotations

import numpy as np

from .geo import point_to_segment_distance_xy
from .road_graph import RoadGraph

WINDOW = 25  # how far ahead/behind the previous match we search first


def match_route_to_track(
    route_arc_ids: np.ndarray, graph: RoadGraph, lons: np.ndarray, lats: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (ground_truth_arc_id, match_dist_m) arrays, one per GPS point.

    ``ground_truth_arc_id`` is -1 if no reasonable match was found (e.g. a
    GPS gap placed the point far from every route segment).
    """
    route_arc_ids = np.asarray(route_arc_ids, dtype=np.int64)
    r_ax, r_ay = graph.ax[route_arc_ids], graph.ay[route_arc_ids]
    r_bx, r_by = graph.bx[route_arc_ids], graph.by[route_arc_ids]

    px, py = graph.projector.project(lons, lats)

    n_route = len(route_arc_ids)
    n_pts = len(lons)
    gt_arc = np.full(n_pts, -1, dtype=np.int64)
    gt_dist = np.full(n_pts, np.nan, dtype=float)

    last_pos = 0
    for i in range(n_pts):
        lo = max(0, last_pos - WINDOW)
        hi = min(n_route, last_pos + WINDOW + 1)
        d = point_to_segment_distance_xy(
            px[i], py[i], r_ax[lo:hi], r_ay[lo:hi], r_bx[lo:hi], r_by[lo:hi]
        )
        if d.size == 0 or np.nanmin(d) > 60.0:
            # Windowed search failed (large GPS gap/jump) -- fall back to a
            # full search over the whole route once.
            d_full = point_to_segment_distance_xy(px[i], py[i], r_ax, r_ay, r_bx, r_by)
            best = int(np.argmin(d_full))
            gt_arc[i] = route_arc_ids[best]
            gt_dist[i] = float(d_full[best])
            last_pos = best
        else:
            best_local = int(np.argmin(d))
            best = lo + best_local
            gt_arc[i] = route_arc_ids[best]
            gt_dist[i] = float(d[best_local])
            last_pos = best

    return gt_arc, gt_dist
