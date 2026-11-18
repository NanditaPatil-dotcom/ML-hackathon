"""Viterbi decoding over per-point road candidates.

States at time t  = candidate road segments for GPS point t.
Emission  P(z_t | r) = probability from the trained model (CatBoost/LGBM/...).
Transition P(r_t -> r_{t+1}) uses the road graph (Newson & Krumm 2009):

    log P = -|route_dist - gps_dist| / beta          (distance continuity)
            - turn_weight * total_turn_deg / 180      (bearing continuity, U-turn = 180)

and is -inf (impossible) when r_{t+1} is not reachable from r_t within
``max_route_factor * gps_dist + max_route_slack`` metres.
"""
import math
from dataclasses import dataclass, field

from .geo import haversine

NEG_INF = float("-inf")


@dataclass
class HMMConfig:
    beta: float = 50.0             # metres; larger = more tolerant of detours
    turn_weight: float = 3.0       # penalty (log units) per 180 degrees of turning along the route
    backtrack_tol: float = 30.0    # metres of backward movement on one road treated as GPS jitter
    max_route_factor: float = 4.0  # route may be at most this x straight-line distance ...
    max_route_slack: float = 300.0  # ... plus this many metres
    min_emission: float = 1e-9     # floor so one zero prob doesn't kill a path
    emission_weight: float = 1.0   # scale of model log-probs vs transition log-probs


@dataclass
class MatchResult:
    road_ids: list                 # best road per GPS point (None for points with no candidates)
    route: list                    # connected route: consecutive-duplicate-free, gaps filled
    log_prob: float
    breaks: list = field(default_factory=list)  # point indices where the HMM had to restart


def _norm_candidates(cands):
    out = []
    for c in cands:
        if isinstance(c, dict):
            out.append((c["road_id"], float(c["emission_prob"])))
        else:
            out.append((c[0], float(c[1])))
    return out


def match_trip(points, graph, config=None):
    """Find the most likely road sequence for one trip.

    points: list (in time order) of either
        {"lat": .., "lon": .., "candidates": [(road_id, emission_prob), ...]}
      or just [(road_id, emission_prob), ...]   (no GPS coords: connectivity-only mode)
    graph:  RoadGraph
    """
    cfg = config or HMMConfig()
    steps = []  # (orig_index, lat, lon, [(road_id, logE, offset)])
    for i, p in enumerate(points):
        if isinstance(p, dict):
            lat, lon, cands = p.get("lat"), p.get("lon"), p["candidates"]
        else:
            lat = lon = None
            cands = p
        states = []
        for rid, prob in _norm_candidates(cands):
            if rid not in graph.seg:
                continue
            off = graph.project(rid, lat, lon) if lat is not None else graph.midpoint_offset(rid)
            states.append((rid, cfg.emission_weight * math.log(max(prob, cfg.min_emission)), off))
        if states:
            steps.append((i, lat, lon, states))

    road_ids = [None] * len(points)
    if not steps:
        return MatchResult(road_ids, [], NEG_INF, [])

    # --- forward pass ---
    score = [s[1] for s in steps[0][3]]
    back = [[None] * len(steps[0][3])]
    gaps = [[None] * len(steps[0][3])]
    breaks = []
    scores = [score]
    for t in range(1, len(steps)):
        _, plat, plon, prev = steps[t - 1]
        _, lat, lon, cur = steps[t]
        if lat is not None and plat is not None:
            gps_d = haversine(plat, plon, lat, lon)
        else:
            gps_d = 0.0
        limit = cfg.max_route_factor * gps_d + cfg.max_route_slack
        new_score, bp, gp = [], [], []
        for rb, eb, ob in cur:
            best, arg, gap = NEG_INF, None, None
            for j, (ra, _, oa) in enumerate(prev):
                if score[j] == NEG_INF:
                    continue
                d, mids, turn = graph.route(ra, oa, rb, ob, max_dist=limit,
                                            backtrack_tol=cfg.backtrack_tol)
                if d == math.inf:
                    continue
                lt = -abs(d - gps_d) / cfg.beta - cfg.turn_weight * turn / 180.0
                s = score[j] + lt
                if s > best:
                    best, arg, gap = s, j, mids
            new_score.append(best + eb if arg is not None else NEG_INF)
            bp.append(arg)
            gp.append(gap)
        if all(s == NEG_INF for s in new_score):
            # HMM break: no candidate reachable -> restart from emissions only
            breaks.append(steps[t][0])
            new_score = [e for _, e, _ in cur]
            bp = [None] * len(cur)
            gp = [None] * len(cur)
        score = new_score
        scores.append(score)
        back.append(bp)
        gaps.append(gp)

    # --- backtrack ---
    j = max(range(len(score)), key=lambda k: score[k])
    total = score[j]
    chosen = [None] * len(steps)
    for t in range(len(steps) - 1, -1, -1):
        chosen[t] = j
        prev_j = back[t][j]
        if prev_j is None and t > 0:
            # start of a segment after a break: continue from the best end state
            # of the previous segment
            prev_j = max(range(len(scores[t - 1])), key=lambda k: scores[t - 1][k])
        j = prev_j

    route = []
    for t, j in enumerate(chosen):
        idx, _, _, states = steps[t]
        rid = states[j][0]
        road_ids[idx] = rid
        mids = gaps[t][j] or []
        for r in mids + [rid]:
            if not route or route[-1] != r:
                route.append(r)
    return MatchResult(road_ids, route, total, breaks)

