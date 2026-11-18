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
for _p in (config.ROOT / "data_processing", config.ROOT / "src", config.ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
try:
    from hmm.api_adapter import viterbi_for_api  # HMM team's decoder
except ImportError:
    viterbi_for_api = None
try:  # preprocessing team's cleaning + candidate/feature code (same code that built the training set)
    from clean import clean_track as dp_clean_track
    from features import build_candidates_table as dp_build_candidates
    from road_graph import RoadGraph as DPRoadGraph
except ImportError:
    dp_clean_track = dp_build_candidates = DPRoadGraph = None
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

    def project(self, lat: float, lon: float, arc_ids=None):
        """Distance (m), snapped lat/lon, and fraction along arc, for every arc (or just arc_ids)."""
        sel = slice(None) if arc_ids is None else np.asarray(arc_ids, dtype=int)
        lon1, lat1, lon2, lat2 = self.lon1[sel], self.lat1[sel], self.lon2[sel], self.lat2[sel]
        kx = math.cos(math.radians(lat)) * EARTH_R * math.pi / 180
        ky = EARTH_R * math.pi / 180
        ax, ay = (lon1 - lon) * kx, (lat1 - lat) * ky
        dx, dy = (lon2 - lon1) * kx, (lat2 - lat1) * ky
        L2 = dx * dx + dy * dy
        t = np.where(L2 > 0, -(ax * dx + ay * dy) / np.where(L2 > 0, L2, 1), 0.0).clip(0, 1)
        px, py = ax + t * dx, ay + t * dy
        dist = np.where(self.valid[sel], np.hypot(px, py), np.inf)
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
    features: Any  # pandas DataFrame: one row per candidate, model feature columns
    scores: np.ndarray = None  # classifier probabilities


# Columns the trained CandidateScorer reads (src/pathsnap/models/base.py NUMERIC_FEATURES).
MODEL_FEATURES = ["dist_to_segment_m", "bearing_alignment_deg", "speed_limit_proxy_mps",
                  "point_speed_mps", "arc_length_m", "candidate_rank"]


def speed_limit_proxy_mps(length_m):
    """Same heuristic as data_processing/road_graph.py (raw data has no speed limits)."""
    return np.clip(8.0 + (np.asarray(length_m) - 30.0) * (22.0 - 8.0) / (300.0 - 30.0), 8.0, 22.0)


class Pipeline:
    def __init__(self):
        self.model = self._load_model(config.MODEL_PATH)
        names = getattr(self.model, "feature_names_in_", None) if self.model is not None else None
        self.feature_names: List[str] = list(names) if names is not None else MODEL_FEATURES

    def status(self) -> Dict[str, str]:
        return {
            "cleaning": "data_processing.clean_track" if dp_clean_track else "builtin speed/duplicate filter",
            "candidates": "data_processing.build_candidates_table" if dp_build_candidates
            else f"builtin radius {config.CANDIDATE_RADIUS_M:.0f} m, top {config.MAX_CANDIDATES}",
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
    def candidates(self, net: RoadNetwork, track: np.ndarray, idx: np.ndarray):
        """Builtin fallback: one row per (point_idx, candidate_arc_id), same columns as
        data_processing.features.build_candidates_table."""
        import pandas as pd

        rows = []
        for pos, i in enumerate(idx):
            lon, lat, t = track[i]
            dist, _, _, _ = net.project(lat, lon)
            order = np.argsort(dist)
            near = order[dist[order] <= config.CANDIDATE_RADIUS_M][: config.MAX_CANDIDATES]
            if len(near) == 0:  # nothing in radius -> keep the closest few so the HMM never breaks
                near = order[: min(3, len(order))]
            j0 = idx[max(pos - 1, 0)]
            j1 = idx[pos] if pos > 0 else idx[min(1, len(idx) - 1)]
            head = float(bearing_deg(track[j0, 1], track[j0, 0], track[j1, 1], track[j1, 0])) if j0 != j1 else 0.0
            dt = track[i, 2] - track[j0, 2]
            speed = float(haversine_m(track[j0, 1], track[j0, 0], lat, lon)) / dt if pos > 0 and dt > 0 else 0.0
            for a in near:
                rows.append({"point_idx": pos, "candidate_arc_id": int(a), "dist_to_segment_m": float(dist[a]),
                             "bearing_alignment_deg": float(angle_diff(head, net.bearing[a])),
                             "arc_length_m": float(net.length[a]),
                             "speed_limit_proxy_mps": float(speed_limit_proxy_mps(net.length[a])),
                             "point_speed_mps": speed})
        return pd.DataFrame(rows)

    def preprocess(self, net: RoadNetwork, track: np.ndarray):
        """-> (keep mask over raw rows, raw row index per cleaned point, candidates DataFrame)."""
        import pandas as pd

        if dp_clean_track is not None:
            try:
                df = pd.DataFrame(track[:, :3], columns=["lon", "lat", "timestamp"])
                df["_raw_idx"] = np.arange(len(df))
                cleaned, _ = dp_clean_track("trip", df)
                raw_idx = cleaned["_raw_idx"].to_numpy(dtype=int)
                g = DPRoadGraph(net.nodes, net.arcs)
                cand = dp_build_candidates("trip", cleaned[["lon", "lat", "timestamp"]].reset_index(drop=True), g)
                keep = np.zeros(len(track), bool)
                keep[raw_idx] = True
                return keep, raw_idx, cand
            except Exception:  # noqa: BLE001
                log.exception("data_processing failed; using builtin preprocessing")
        keep = self.clean(track)
        raw_idx = np.flatnonzero(keep)
        return keep, raw_idx, self.candidates(net, track, raw_idx)

    def to_layers(self, net: RoadNetwork, track: np.ndarray, raw_idx: np.ndarray, cand) -> List[Layer]:
        if cand is None or len(cand) == 0:
            return []
        cand = cand.copy()
        cand["trip_id"] = "trip"
        cand["candidate_rank"] = cand.groupby("point_idx")["dist_to_segment_m"].rank(method="first").astype(int) - 1
        layers = []
        for pidx, g in cand.groupby("point_idx", sort=True):
            ri = int(raw_idx[int(pidx)])
            arc_ids = g["candidate_arc_id"].to_numpy(dtype=int)
            _, slat, slon, _ = net.project(track[ri, 1], track[ri, 0], arc_ids)
            layers.append(Layer(ri, arc_ids, g["dist_to_segment_m"].to_numpy(float),
                                np.column_stack([slon, slat]), g))
        return layers

    # -- stage 3: classifier scoring -----------------------------------------
    # INTEGRATION POINT (model team): drop final_model.joblib into models/ (any sklearn-style
    # estimator/pipeline with predict_proba + feature_names_in_). No code change needed.
    def score(self, layers: List[Layer]) -> None:
        if self.model is not None and layers:
            try:
                import pandas as pd

                df = pd.concat([L.features for L in layers], ignore_index=True)
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
            b = L.features["bearing_alignment_deg"].to_numpy(float)
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
        per_obs_hmm: List[Optional[int]] = [None] * len(track)
        per_obs_clf: List[Optional[int]] = [None] * len(track)
        keep = np.zeros(len(track), bool)
        if net.n_arcs and len(track):
            keep, raw_idx, cand = self.preprocess(net, track)
            layers = self.to_layers(net, track, raw_idx, cand)
            if layers:
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
