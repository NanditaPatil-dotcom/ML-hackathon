"""Road network graph used for HMM transition probabilities.

A road *segment* (road_id) goes from node ``u`` to node ``v``. If it is not
``oneway`` it can be driven in both directions. Positions on a segment are
expressed as an *offset* in metres from ``u`` along its geometry.
"""
import heapq
import json
import math

from .geo import angle_diff, bearing, haversine, project_on_polyline

INF = float("inf")


class RoadGraph:
    def __init__(self, nodes, segments):
        """
        nodes:    {node_id: (lat, lon)}
        segments: iterable of dicts with keys
                  road_id, u, v, [oneway=False], [geometry=[(lat, lon), ...]], [length=metres]
        """
        self.nodes = {k: tuple(v) for k, v in nodes.items()}
        self.seg = {}
        self.adj = {n: [] for n in self.nodes}  # node -> [(next_node, length, road_id)]
        for s in segments:
            rid, u, v = s["road_id"], s["u"], s["v"]
            geom = [tuple(p) for p in s.get("geometry") or [self.nodes[u], self.nodes[v]]]
            length = s.get("length")
            if length is None:
                length = sum(haversine(*a, *b) for a, b in zip(geom[:-1], geom[1:]))
            oneway = bool(s.get("oneway", False))
            self.seg[rid] = {
                "u": u, "v": v, "oneway": oneway, "geometry": geom, "length": float(length),
                "bearing": bearing(*geom[0], *geom[-1]),
            }
            self.adj.setdefault(u, []).append((v, float(length), rid))
            if not oneway:
                self.adj.setdefault(v, []).append((u, float(length), rid))
        self._sp_cache = {}

    # ---------- loading ----------
    @classmethod
    def from_dict(cls, d):
        return cls(d["nodes"], d["segments"])

    @classmethod
    def from_arrays(cls, nodes, arcs):
        """PathSnap .nodes (n,2 lon,lat) + .arcs (m,2 from,to); arc_id = row, arcs are directed."""
        n = len(nodes)
        node_map = {i: (float(nodes[i][1]), float(nodes[i][0])) for i in range(n)}
        segs = []
        for aid, (u, v) in enumerate(arcs):
            u, v = int(u), int(v)
            if 0 <= u < n and 0 <= v < n:
                segs.append({"road_id": aid, "u": u, "v": v, "oneway": True})
        return cls(node_map, segs)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))

    # ---------- geometry ----------
    def project(self, road_id, lat, lon):
        """Offset (m from u) of the GPS point's projection onto the segment."""
        off, _, _ = project_on_polyline(lat, lon, self.seg[road_id]["geometry"])
        return off

    def midpoint_offset(self, road_id):
        return self.seg[road_id]["length"] / 2.0

    # ---------- shortest paths ----------
    def _dijkstra(self, src, max_dist):
        key = (src, max_dist)
        if key in self._sp_cache:
            return self._sp_cache[key]
        dist, pred = {src: 0.0}, {}
        pq = [(0.0, src)]
        while pq:
            d, n = heapq.heappop(pq)
            if d > dist.get(n, INF) or d > max_dist:
                continue
            for m, w, rid in self.adj.get(n, ()):
                nd = d + w
                if nd < dist.get(m, INF) and nd <= max_dist:
                    dist[m], pred[m] = nd, (n, rid)
                    heapq.heappush(pq, (nd, m))
        self._sp_cache[key] = (dist, pred)
        return dist, pred

    @staticmethod
    def _path(pred, src, dst):
        """[(road_id, from_node, to_node), ...] from src to dst."""
        out, n = [], dst
        while n != src:
            p, rid = pred[n]
            out.append((rid, p, n))
            n = p
        return out[::-1]

    def heading(self, road_id, from_node):
        """Travel heading on a segment when entering it at from_node."""
        s = self.seg[road_id]
        return s["bearing"] if from_node == s["u"] else (s["bearing"] + 180.0) % 360.0

    def _turns(self, headings):
        return sum(angle_diff(a, b) for a, b in zip(headings[:-1], headings[1:]))

    def route(self, ra, off_a, rb, off_b, max_dist=2000.0, backtrack_tol=0.0):
        """Shortest drivable route from (ra, off_a) to (rb, off_b).

        Returns (distance_m, intermediate_road_ids, total_turn_deg).
        distance is INF if unreachable within max_dist. Turns include a
        180-degree U-turn when the route reverses direction.
        backtrack_tol: on the same segment, backward moves up to this many
        metres are treated as GPS jitter (distance 0, no U-turn).
        """
        A, B = self.seg[ra], self.seg[rb]
        best = (INF, [], 0.0)
        if ra == rb:
            if off_b >= off_a - backtrack_tol:
                best = (max(0.0, off_b - off_a), [], 0.0)
            elif not A["oneway"]:
                best = (off_a - off_b, [], 0.0)  # driving the two-way road the other way
        # leave A via its end node(s): (node, cost, heading on A)
        exits = [(A["v"], A["length"] - off_a, A["bearing"])]
        if not A["oneway"]:
            exits.append((A["u"], off_a, (A["bearing"] + 180.0) % 360.0))
        # enter B via its start node(s): (node, cost, heading on B)
        entries = [(B["u"], off_b, B["bearing"])]
        if not B["oneway"]:
            entries.append((B["v"], B["length"] - off_b, (B["bearing"] + 180.0) % 360.0))
        for xn, xc, xh in exits:
            if xc >= best[0]:
                continue
            dist, pred = self._dijkstra(xn, max_dist)
            for en, ec, eh in entries:
                if en not in dist:
                    continue
                total = xc + dist[en] + ec
                if total < best[0]:
                    path = self._path(pred, xn, en)
                    hs = [xh] + [self.heading(r, a) for r, a, _ in path] + [eh]
                    best = (total, [r for r, _, _ in path], self._turns(hs))
        if best[0] > max_dist:
            return INF, [], 0.0
        return best
