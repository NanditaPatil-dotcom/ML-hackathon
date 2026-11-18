# HMM / Viterbi route reconstruction

Turns per-point candidate arcs (with classifier probabilities as emissions) into the most likely connected route for a trip.

```python
from pathsnap.decoder import HMMRouteDecoder
dec = HMMRouteDecoder(arcs, nodes=nodes)          # PathSnap .arcs (from,to) and .nodes (lon,lat)
best_arc_per_obs = dec.decode(cands_dict, track)  # cands_dict[t] = [{"arc_id", "score"[, "projection_ratio"]}]
```

Without `nodes` it falls back to the old topology-only heuristic.

- Emission: model `score` (floored at 1e-9).
- Transition (Newson & Krumm): `-|route_dist - gps_dist|/beta - turn_weight * turn_deg/180`. `route_dist` is the shortest drivable distance (Dijkstra, directed arcs) between the two candidate positions. Unreachable within `4*gps_dist+300 m` is impossible.
- Viterbi in log space; restarts after a break; `MatchResult.route` fills gaps with shortest-path arcs.
- Defaults (`HMMConfig`): beta 50 m, turn_weight 3, backtrack_tol 30 m.

On a mock 4x4 grid at 25 m noise: top-1 candidate 73.5% -> Viterbi 85.2% (see tests/test_hmm_viterbi.py).
