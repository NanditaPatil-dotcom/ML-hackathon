"""Road-network graph + spatial index for one trip's .nodes/.arcs files.

Each trip ships its own regional road network (tens of thousands of arcs).
We build:
  - per-arc geometry (endpoint coords, length, bearing),
  - a KD-tree over arc midpoints (in a local metric projection) for fast
    approximate nearest-segment lookup,
  - node->outgoing-arc and node->incoming-arc adjacency, which the HMM team
    needs to know which arcs are reachable from which (two arcs are
    road-network-connected if one's to_node equals the other's from_node).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from geo import LocalProjector, bearing_deg, haversine_m, point_to_segment_distance_xy

# A candidate arc's "speed limit proxy": the raw data has no explicit speed
# limit field, so we proxy it from segment length, which correlates with
# road class in most OSM-derived networks (short segments cluster around
# intersections/residential streets; long, straight segments are arterials
# or highways). This is an explicit, documented assumption -- not a ground
# truth speed limit -- and is reported as such in the technical report.
def _speed_limit_proxy_mps(length_m: np.ndarray) -> np.ndarray:
    # Piecewise-linear heuristic, clipped to a plausible urban/suburban range.
    # <30m: residential (~8 m/s / 30km/h) .. >300m: arterial (~22 m/s / 80km/h)
    proxy = 8.0 + (length_m - 30.0) * (22.0 - 8.0) / (300.0 - 30.0)
    return np.clip(proxy, 8.0, 22.0)


@dataclass
class CandidateArc:
    arc_id: int
    dist_to_segment_m: float
    arc_bearing_deg: float
    length_m: float
    speed_limit_proxy_mps: float


class RoadGraph:
    def __init__(self, nodes: np.ndarray, arcs: np.ndarray):
        self.nodes = nodes  # (N, 2) lon, lat
        self.arcs = arcs  # (M, 2) from_node, to_node
        self.n_arcs = len(arcs)

        origin_lon, origin_lat = nodes[:, 0].mean(), nodes[:, 1].mean()
        self.projector = LocalProjector(origin_lon, origin_lat)

        from_coords = nodes[arcs[:, 0]]
        to_coords = nodes[arcs[:, 1]]
        self.from_lonlat = from_coords
        self.to_lonlat = to_coords

        self.ax, self.ay = self.projector.project(from_coords[:, 0], from_coords[:, 1])
        self.bx, self.by = self.projector.project(to_coords[:, 0], to_coords[:, 1])
        self.mid_x = (self.ax + self.bx) / 2.0
        self.mid_y = (self.ay + self.by) / 2.0

        self.length_m = haversine_m(
            from_coords[:, 0], from_coords[:, 1], to_coords[:, 0], to_coords[:, 1]
        )
        self.bearing_deg = bearing_deg(
            from_coords[:, 0], from_coords[:, 1], to_coords[:, 0], to_coords[:, 1]
        )
        self.speed_limit_proxy_mps = _speed_limit_proxy_mps(self.length_m)

        self._tree = cKDTree(np.column_stack([self.mid_x, self.mid_y]))

        # node_id -> list of arc_ids starting there / ending there (for HMM
        # transition connectivity: arc i -> arc j is a valid transition iff
        # arcs[i, 1] (to_node) == arcs[j, 0] (from_node)).
        self.out_arcs_by_node: dict[int, list[int]] = {}
        for arc_id, from_node in enumerate(arcs[:, 0]):
            self.out_arcs_by_node.setdefault(int(from_node), []).append(arc_id)

    def arc_transitions(self, arc_id: int) -> list[int]:
        """Arc ids reachable in one hop from ``arc_id`` (shares its to_node)."""
        to_node = int(self.arcs[arc_id, 1])
        return self.out_arcs_by_node.get(to_node, [])

    def query_candidates(
        self, lon: float, lat: float, radius_m: float, k: int = 30
    ) -> list[CandidateArc]:
        """Candidate road segments within ``radius_m`` of (lon, lat).

        Two-stage search: first take the ``k`` nearest arcs by midpoint
        distance (cheap KD-tree query), then compute the exact
        perpendicular point-to-segment distance for just those ``k`` and
        keep the ones within ``radius_m``. This avoids computing exact
        distances against all (tens of thousands of) arcs per point.
        """
        px, py = self.projector.project(lon, lat)
        k_eff = min(k, self.n_arcs)
        _, idx = self._tree.query([px, py], k=k_eff)
        idx = np.atleast_1d(idx)

        dists = point_to_segment_distance_xy(
            px, py, self.ax[idx], self.ay[idx], self.bx[idx], self.by[idx]
        )
        out = []
        for local_i, arc_id in enumerate(idx):
            d = float(dists[local_i])
            if d <= radius_m:
                out.append(
                    CandidateArc(
                        arc_id=int(arc_id),
                        dist_to_segment_m=d,
                        arc_bearing_deg=float(self.bearing_deg[arc_id]),
                        length_m=float(self.length_m[arc_id]),
                        speed_limit_proxy_mps=float(self.speed_limit_proxy_mps[arc_id]),
                    )
                )
        out.sort(key=lambda c: c.dist_to_segment_m)
        return out
