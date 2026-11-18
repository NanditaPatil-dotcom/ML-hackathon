"""PathSnap inference pipeline (runs on Laptop 1):

    .track -> clean -> candidate arcs -> classifier scores -> HMM/Viterbi -> arc sequence

Every stage is a method with a working default so the API is demoable today.
Teammates' real code replaces a stage at the marked INTEGRATION POINTs; keep the
same inputs/outputs and the API needs no other changes.
"""
from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import config

log = logging.getLogger("pathsnap.pipeline")
# Teammates' code: src/ (candidate_scorer, needed to unpickle the model) and hmm/ (Viterbi).
for _p in (config.ROOT / "src", config.ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
try:
    from hmm.api_adapter import viterbi_for_api  # HMM team's decoder
except ImportError:
    viterbi_for_api = None
EARTH_R = 6_371_000.0


# --------------------------------------------------------------------------- geo helpers
def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def angle_diff(a, b):
    d = np.abs(np.asarray(a) - np.asarray(b)) % 360
    return np.minimum(d, 360 - d)


# --------------------------------------------------------------------------- road network
class RoadNetwork:
    """Directed arcs between nodes; arc_id = row index in .arcs."""

    def __init__(self, nodes: np.ndarray, arcs: np.ndarray):
        self.nodes, self.arcs = nodes, arcs
        self.n_arcs = len(arcs)
        valid = (arcs >= 0).all(axis=1) & (arcs < len(nodes)).all(axis=1) if len(arcs) else np.zeros(0, bool)
        self.valid = valid
        a = np.where(valid[:, None], arcs, 0) if len(arcs) else arcs
        self.lon1, self.lat1 = nodes[a[:, 0], 0], nodes[a[:, 0], 1]
        self.lon2, self.lat2 = nodes[a[:, 1], 0], nodes[a[:, 1], 1]
        self.bearing = bearing_deg(self.lat1, self.lon1, self.lat2, self.lon2)
        self.length = haversine_m(self.lat1, self.lon1, self.lat2, self.lon2)

    def project(self, lat: float, lon: float):
        """Distance (m), snapped lat/lon, and fraction along arc for every arc."""
        kx = math.cos(math.radians(lat)) * EARTH_R * math.pi / 180
        ky = EARTH_R * math.pi / 180
        ax, ay = (self.lon1 - lon) * kx, (self.lat1 - lat) * ky
        dx, dy = (self.lon2 - self.lon1) * kx, (self.lat2 - self.lat1) * ky
        L2 = dx * dx + dy * dy
        t = np.where(L2 > 0, -(ax * dx + ay * dy) / np.where(L2 > 0, L2, 1), 0.0).clip(0, 1)
        px, py = ax + t * dx, ay + t * dy
        dist = np.where(self.valid, np.hypot(px, py), np.inf)
        return dist, lat + py / ky, lon + px / kx, t

    def transition_cost(self, i: int, j: int) -> float:
        """Topology cost (in log-prob units) of moving from arc i to arc j between two fixes."""
        if i == j:
            return 0.0
        fi, ti = self.arcs[i]
        fj, tj = self.arcs[j]
        if fj == ti and tj == fi:
            return 2.0  # U-turn onto the reverse arc
        if fj == ti:
            return 0.0  # j directly follows i
        if ti == tj or fi == fj or fi == tj:
            return 1.0  # touches i but against arc direction
        return config.UNCONNECTED_PENALTY

    def arc_coords(self, arc_ids: Sequence[int]) -> List[List[float]]:
        """[[lon, lat], ...] polyline through the given arcs (consecutive duplicates collapsed)."""
        out: List[List[float]] = []
        for a in arc_ids:
            if a is None or a < 0 or a >= self.n_arcs:
                continue
            for n in self.arcs[a]:
                pt = [float(self.nodes[n, 0]), float(self.nodes[n, 1])]
                if not out or out[-1] != pt:
                    out.append(pt)
        return out


# --------------------------------------------------------------------------- pipeline
@dataclass
class Layer:
    """Candidates for one GPS observation."""
    obs_index: int
    arc_ids: np.ndarray
    dist_m: np.ndarray
    snapped: np.ndarray  # (k, 2) lon, lat
    features: Dict[str, np.ndarray]  # feature name -> (k,) values
    scores: np.ndarray = None  # classifier probabilities


# Columns the model team trained on (src/candidate_scorer.py):
#   distance_to_segment (m), bearing_diff (deg 0-180), speed_limit (km/h),
#   transition_distance (m), candidate_road_id (arc_id)
MODEL_FEATURES = ["distance_to_segment", "bearing_diff", "speed_limit", "transition_distance", "candidate_road_id"]


class Pipeline:
    def __init__(self):
        self.model = self._load_model(config.MODEL_PATH)
        names = getattr(self.model, "feature_names_in_", None) if self.model is not None else None
        self.feature_names: List[str] = list(names) if names is not None else MODEL_FEATURES

    def status(self) -> Dict[str, str]:
        return {
            "cleaning": "builtin speed/duplicate filter",
            "candidates": f"builtin radius {config.CANDIDATE_RADIUS_M:.0f} m, top {config.MAX_CANDIDATES}",
            "scoring": f"model {getattr(self.model, 'model_name', type(self.model).__name__)} ({config.MODEL_PATH.name})"
            if self.model is not None
            else "STUB heuristic (no trained model loaded)",
            "route_reconstruction": "hmm.api_adapter (Newson-Krumm HMM/Viterbi)" if viterbi_for_api
            else "builtin HMM/Viterbi (connectivity-aware)",
        }

    @staticmethod
    def _load_model(path: Path):
        if not path.exists():
            log.warning("Model not found at %s -> heuristic scoring", path)
            return None
        try:
            import joblib

            m = joblib.load(path)
            log.info("Loaded model %s from %s", type(m).__name__, path)
            return m
        except Exception:  # noqa: BLE001 - keep the server up for the demo
            log.exception("Failed to load model %s -> heuristic scoring", path)
            return None

    # -- stage 1: cleaning ----------------------------------------------------
    # INTEGRATION POINT (preprocessing): track (n,3 lon,lat,t) -> boolean keep mask.
    def clean(self, track: np.ndarray) -> np.ndarray:
        n = len(track)
        keep = np.isfinite(track).all(axis=1) & (np.abs(track[:, 1]) <= 90) & (np.abs(track[:, 0]) <= 180)
        last = None
        for i in range(n):
            if not keep[i]:
                continue
            if last is not None:
                d = float(haversine_m(track[last, 1], track[last, 0], track[i, 1], track[i, 0]))
                dt = track[i, 2] - track[last, 2]
                if dt <= 0 and d < 1:  # duplicate ping / non-increasing time
                    keep[i] = False
                    continue
                if dt > 0 and d / dt > config.MAX_SPEED_MPS:  # teleport
                    keep[i] = False
                    continue
            last = i
        return keep

    # -- stage 2: candidates + features -------------------------------------
    # INTEGRATION POINT (preprocessing): must produce the SAME feature columns the model was trained on.
    def candidates(self, net: RoadNetwork, track: np.ndarray, idx: np.ndarray) -> List[Layer]:
        layers = []
        for pos, i in enumerate(idx):
            lon, lat, t = track[i]
            dist, slat, slon, frac = net.project(lat, lon)
            order = np.argsort(dist)
            near = order[dist[order] <= config.CANDIDATE_RADIUS_M][: config.MAX_CANDIDATES]
            if len(near) == 0:  # nothing in radius -> keep the closest few so the HMM never breaks
                near = order[: min(3, len(order))]
            # heading / speed from neighbouring kept observations
            j0, j1 = idx[max(pos - 1, 0)], idx[min(pos + 1, len(idx) - 1)]
            if j0 != j1:
                head = float(bearing_deg(track[j0, 1], track[j0, 0], track[j1, 1], track[j1, 0]))
                dt = max(track[j1, 2] - track[j0, 2], 1e-6)
                speed = float(haversine_m(track[j0, 1], track[j0, 0], track[j1, 1], track[j1, 0])) / dt
            else:
                head, speed = None, 0.0
            bdiff = np.zeros(len(near)) if head is None else angle_diff(head, net.bearing[near])
            snapped = np.column_stack([slon[near], slat[near]])
            if layers:  # straight-line distance from the previous point's closest candidate
                prev = layers[-1].snapped[0]
                trans = haversine_m(prev[1], prev[0], snapped[:, 1], snapped[:, 0])
            else:
                trans = np.zeros(len(near))
            feats = {
                "distance_to_segment": dist[near],
                "bearing_diff": bdiff,
                "speed_limit": np.full(len(near), config.DEFAULT_SPEED_LIMIT_KMH),
                "transition_distance": trans,
                "candidate_road_id": near.astype(int),
                "speed_mps": np.full(len(near), speed),
            }
            layers.append(Layer(int(i), near.astype(int), dist[near], snapped, feats))
        return layers

    # -- stage 3: classifier scoring -----------------------------------------
    # INTEGRATION POINT (model team): drop final_model.joblib into models/ (any sklearn-style
    # estimator/pipeline with predict_proba + feature_names_in_). No code change needed.
    def score(self, layers: List[Layer]) -> None:
        if self.model is not None and layers:
            try:
                import pandas as pd

                cols = {f: np.concatenate([L.features.get(f, np.zeros(len(L.arc_ids))) for L in layers])
                        for f in self.feature_names}
                cols["point_id"] = np.concatenate([np.full(len(L.arc_ids), L.obs_index) for L in layers])
                df = pd.DataFrame(cols)
                if hasattr(self.model, "emission_log_prob"):  # model team's CandidateScorer wrapper
                    proba = np.asarray(self.model.predict_proba(df))
                else:  # plain sklearn-style estimator
                    proba = self.model.predict_proba(df[self.feature_names])[:, -1]
                ofs = 0
                for L in layers:  # one batched call per trip, split back per GPS point
                    L.scores = proba[ofs: ofs + len(L.arc_ids)]
                    ofs += len(L.arc_ids)
                return
            except Exception:  # noqa: BLE001
                log.exception("Model scoring failed; falling back to heuristic")
        for L in layers:  # Gaussian GPS-noise likelihood x heading agreement
            e = np.exp(-0.5 * (L.dist_m / config.GPS_SIGMA_M) ** 2)
            b = L.features["bearing_diff"]
            L.scores = e * (0.7 + 0.3 * (0.5 + 0.5 * np.cos(np.radians(b))))

    # -- stage 4: HMM / Viterbi ---------------------------------------------
    # INTEGRATION POINT (HMM team): layers (with .scores as emissions) -> chosen candidate index per layer.
    def viterbi(self, net: RoadNetwork, track: np.ndarray, layers: List[Layer]) -> List[int]:
        if viterbi_for_api is not None:
            try:
                return viterbi_for_api(net, track, layers)
            except Exception:  # noqa: BLE001
                log.exception("hmm.api_adapter failed; using builtin Viterbi")
        eps = 1e-9
        lp = np.log(layers[0].scores + eps)
        back = []
        for t in range(1, len(layers)):
            A_, B_ = layers[t - 1], layers[t]
            pa, pb = track[A_.obs_index], track[B_.obs_index]
            gc = float(haversine_m(pa[1], pa[0], pb[1], pb[0]))
            sd = haversine_m(A_.snapped[:, None, 1], A_.snapped[:, None, 0], B_.snapped[None, :, 1], B_.snapped[None, :, 0])
            trans = -np.abs(sd - gc) / config.TRANSITION_BETA_M
            gap = pb[2] - pa[2] > 10  # long GPS gap: relax connectivity requirement
            if not gap:
                trans = trans - np.array([[net.transition_cost(a, b) for b in B_.arc_ids] for a in A_.arc_ids])
            tot = lp[:, None] + trans
            back.append(tot.argmax(axis=0))
            lp = tot.max(axis=0) + np.log(B_.scores + eps)
        path = [int(lp.argmax())]
        for bp in reversed(back):
            path.append(int(bp[path[-1]]))
        return path[::-1]

    # -- run -------------------------------------------------------------------
    def run(self, track: np.ndarray, nodes: np.ndarray, arcs: np.ndarray) -> Dict[str, Any]:
        net = RoadNetwork(nodes, arcs)
        keep = self.clean(track)
        idx = np.flatnonzero(keep)
        per_obs_hmm: List[Optional[int]] = [None] * len(track)
        per_obs_clf: List[Optional[int]] = [None] * len(track)
        if len(idx) and net.n_arcs:
            layers = self.candidates(net, track, idx)
            self.score(layers)
            choice = self.viterbi(net, track, layers)
            for L, k in zip(layers, choice):
                per_obs_hmm[L.obs_index] = int(L.arc_ids[k])
                per_obs_clf[L.obs_index] = int(L.arc_ids[int(np.argmax(L.scores))])
        return {
            "network": net,
            "kept_mask": keep,
            "per_observation_arcs": per_obs_hmm,
            "classifier_only_arcs": per_obs_clf,
            "route_hmm": collapse(per_obs_hmm),
            "route_clf": collapse(per_obs_clf),
        }


def collapse(seq: Sequence[Optional[int]]) -> List[int]:
    out: List[int] = []
    for a in seq:
        if a is not None and (not out or out[-1] != a):
            out.append(a)
    return out
