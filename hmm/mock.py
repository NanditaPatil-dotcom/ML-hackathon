"""Synthetic toy city + noisy trip, used until real preprocessing/model output lands.

Grid of 4x4 intersections ~150 m apart. Horizontal road ids look like
"H<row>_<col>", vertical ones "V<row>_<col>".
"""
import math
import random

from .geo import haversine, project_on_polyline
from .graph import RoadGraph

LAT0, LON0 = 12.9716, 77.5946   # Bengaluru-ish
STEP = 0.00135                  # ~150 m


def build_mock_graph(n=4):
    nodes = {f"n{r}{c}": (LAT0 + r * STEP, LON0 + c * STEP) for r in range(n) for c in range(n)}
    segs = []
    for r in range(n):
        for c in range(n - 1):
            segs.append({"road_id": f"H{r}_{c}", "u": f"n{r}{c}", "v": f"n{r}{c+1}"})
    for r in range(n - 1):
        for c in range(n):
            segs.append({"road_id": f"V{r}_{c}", "u": f"n{r}{c}", "v": f"n{r+1}{c}"})
    # make one road one-way (west -> east) to exercise direction handling
    for s in segs:
        if s["road_id"] == "H2_1":
            s["oneway"] = True
    return {"nodes": nodes, "segments": segs}


# True route: east along row 0, north up column 2, east along row 2.
TRUE_ROUTE = ["H0_0", "H0_1", "V0_2", "V1_2", "H2_2"]


def _interp(geom, frac):
    (a, b), (c, d) = geom[0], geom[-1]
    return a + frac * (c - a), b + frac * (d - b)


def candidates_for_point(graph, lat, lon, radius=120.0, sigma=25.0, k=4):
    """Toy stand-in for preprocessing + model: nearest segments, Gaussian emission."""
    out = []
    for rid, s in graph.seg.items():
        _, _, (plat, plon) = project_on_polyline(lat, lon, s["geometry"])
        d = haversine(lat, lon, plat, plon)
        if d <= radius:
            out.append((rid, math.exp(-0.5 * (d / sigma) ** 2)))
    out.sort(key=lambda x: -x[1])
    out = out[:k]
    z = sum(p for _, p in out) or 1.0
    return [(rid, p / z) for rid, p in out]


def simulate_trip(graph, route=TRUE_ROUTE, pts_per_seg=3, noise_m=30.0, seed=7):
    """Returns (points, truth). points = [{lat, lon, candidates}], truth = road per point."""
    rng = random.Random(seed)
    deg = noise_m / 111000.0
    points, truth = [], []
    for rid in route:
        s = graph.seg[rid]
        for k in range(pts_per_seg):
            lat, lon = _interp(s["geometry"], (k + 0.5) / pts_per_seg)
            lat += rng.gauss(0, deg)
            lon += rng.gauss(0, deg)
            points.append({"lat": lat, "lon": lon, "candidates": candidates_for_point(graph, lat, lon)})
            truth.append(rid)
    return points, truth
