"""Scorer-driven route reconstruction: preprocessing -> CatBoost -> HMM/Viterbi.

Unlike ``reconstruct_route_for_trip`` (which builds the 9 legacy features for
the old random forest), this path uses the same code that trained the model:

    load_trip -> clean_track -> RoadGraph -> build_candidates_table
      -> add_candidate_rank -> CandidateScorer (the selected model)
      -> HMMRouteDecoder (distance-aware Viterbi)

    from pathsnap.models.scorer import CandidateScorer
    scorer = CandidateScorer.load("artifacts/models/catboost_best.joblib")
    res = reconstruct_trip_with_scorer("Student folder/student_test", "00000009", scorer)
"""
from __future__ import annotations

import numpy as np

from pathsnap.data.build_features import add_candidate_rank
from pathsnap.data.preprocessing.clean import clean_track
from pathsnap.data.preprocessing.features import build_candidates_table
from pathsnap.data.preprocessing.io_utils import load_trip
from pathsnap.data.preprocessing.road_graph import RoadGraph
from pathsnap.data.preprocessing.route_match import match_route_to_track

from .hmm_viterbi import HMMRouteDecoder


def _continuity(path, arcs):
    if len(path) <= 1:
        return 100.0
    ok = sum(path[i] == path[i - 1] or arcs[path[i - 1]][1] == arcs[path[i]][0]
             for i in range(1, len(path)))
    return float(ok / (len(path) - 1) * 100.0)


def reconstruct_trip_with_scorer(partition_dir, trip_id, scorer, radius_m=40.0, k=30):
    """Run the full pipeline on one trip folder (``<partition_dir>/<trip_id>/``).

    Returns a dict with per-point ``clf_path`` / ``hmm_path`` (arc id per point
    that has candidates, keyed by point index), the clean track, and metrics.
    Accuracy is over all clean points that have a ground-truth arc; a point
    with no candidates counts as wrong.
    """
    raw = load_trip(partition_dir, trip_id)
    clean_df, _ = clean_track(trip_id, raw.track)
    if len(clean_df) == 0:
        return None
    graph = RoadGraph(raw.nodes, raw.arcs)

    gt = None
    if raw.route is not None and len(raw.route) > 0:
        gt, _ = match_route_to_track(
            raw.route, graph, clean_df["lon"].to_numpy(), clean_df["lat"].to_numpy()
        )

    cand = build_candidates_table(trip_id, clean_df, graph, gt, radius_m=radius_m, k=k)
    if len(cand) == 0:
        return None
    cand = add_candidate_rank(cand)
    cand["emission"] = np.exp(scorer.emission_log_prob(cand))  # normalised per point

    cands_per_obs = {}
    for pidx, arc, p in zip(cand["point_idx"].to_numpy(), cand["candidate_arc_id"].to_numpy(),
                            cand["emission"].to_numpy()):
        cands_per_obs.setdefault(int(pidx), []).append({"arc_id": int(arc), "score": float(p)})

    obs = sorted(cands_per_obs)
    clf_path = {t: max(cands_per_obs[t], key=lambda c: c["score"])["arc_id"] for t in obs}
    track = clean_df[["lon", "lat"]].to_numpy()
    decoder = HMMRouteDecoder(raw.arcs, nodes=raw.nodes)
    hmm_path = dict(zip(obs, decoder.decode(cands_per_obs, track=track)))

    metrics = {
        "n_points": int(len(clean_df)),
        "n_points_with_candidates": len(obs),
        "clf_continuity": _continuity([clf_path[t] for t in obs], raw.arcs),
        "hmm_continuity": _continuity([hmm_path[t] for t in obs], raw.arcs),
    }
    if gt is not None:
        eval_pts = [i for i in range(len(clean_df)) if gt[i] >= 0]
        n = max(len(eval_pts), 1)
        metrics["clf_acc"] = 100.0 * sum(clf_path.get(i) == gt[i] for i in eval_pts) / n
        metrics["hmm_acc"] = 100.0 * sum(hmm_path.get(i) == gt[i] for i in eval_pts) / n
        on = [i for i in eval_pts if i in clf_path]
        m = max(len(on), 1)
        metrics["clf_acc_on_candidates"] = 100.0 * sum(clf_path[i] == gt[i] for i in on) / m
        metrics["hmm_acc_on_candidates"] = 100.0 * sum(hmm_path[i] == gt[i] for i in on) / m
        metrics["n_eval_points"] = len(eval_pts)
    return {"trip_id": trip_id, "track": clean_df, "arcs": raw.arcs, "nodes": raw.nodes,
            "route": raw.route, "clf_path": clf_path, "hmm_path": hmm_path, "metrics": metrics}
