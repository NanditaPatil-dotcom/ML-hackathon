# HMM / Viterbi route reconstruction

Turns per-point road candidates (with emission probabilities from the trained
model) into the single most likely, **connected** route for each trip.

Pure Python standard library, no extra dependencies.

## How it works

* **Hidden states**: at GPS point *t*, the candidate road segments for that point.
* **Emission**: `log P(gps_t | road)` = log of the model's `emission_prob`
  (floored at `1e-9`, optionally scaled by `emission_weight`).
* **Transition** between road `a` at point *t* and road `b` at point *t+1*
  (Newson & Krumm 2009 style), using the road graph:
  * Project both GPS points onto their candidate segments.
  * `route_dist` = shortest *drivable* distance (Dijkstra, respects `oneway`)
    from the projection on `a` to the projection on `b`.
  * `gps_dist` = straight-line distance between the two GPS points.
  * `log P = -|route_dist - gps_dist| / beta - turn_weight * total_turn_deg / 180`
    (distance continuity + bearing continuity; a U-turn costs 180 deg).
  * Impossible (`-inf`) if `b` isn't reachable from `a` within
    `4 * gps_dist + 300 m`.
* **Viterbi** in log space picks the best road per point. Gaps between
  non-adjacent roads are filled with the shortest-path roads, so `route` is a
  continuous road sequence.
* **Breaks**: if no candidate at point *t* is reachable from any candidate at
  *t-1* (e.g. GPS jump), the HMM restarts at *t*; the index is reported in `breaks`.

## Python API

```python
from hmm import RoadGraph, match_trip
from hmm.matcher import HMMConfig

graph = RoadGraph(nodes={"n1": (lat, lon), ...},
                  segments=[{"road_id": "r1", "u": "n1", "v": "n2", "oneway": False}, ...])
points = [
    {"lat": 12.97, "lon": 77.59, "candidates": [("r1", 0.8), ("r7", 0.2)]},
    ...
]                                 # in time order
res = match_trip(points, graph, HMMConfig(beta=50, turn_weight=3))
res.road_ids   # best road per point (None if a point had no candidates)
res.route      # connected road sequence for the trip
res.breaks     # point indices where the chain had to restart
```

Points may also be bare candidate lists `[("r1", 0.8), ...]` (no lat/lon);
the HMM then uses connectivity only (shorter routes preferred).

## File interface (what teammates need to produce)

```bash
python -m hmm.run --graph roads.csv --candidates candidates.csv --out matched.csv
```

**Preprocessing -> `roads.csv`** (one row per road segment):

| column | meaning |
|---|---|
| `road_id` | unique segment id (same ids the model scores) |
| `u`, `v` | start / end node ids (intersections) |
| `u_lat,u_lon,v_lat,v_lon` | node coordinates |
| `oneway` | optional, 1 = only drivable u -> v (default two-way) |
| `length_m` | optional, else computed |
| `geometry` | optional polyline `"lat lon;lat lon;..."` incl. endpoints, for curved roads |

Two segments are connected iff they share a node id. A `roads.json`
(`{"nodes": {...}, "segments": [...]}`) also works.

**Model -> `candidates.csv`** (one row per candidate per GPS point):

| column | meaning |
|---|---|
| `trip_id` | trip identifier |
| `point_idx` | order of the GPS point within the trip (time order) |
| `lat`, `lon` | raw GPS coordinates of the point |
| `road_id` | candidate segment (must exist in `roads.csv`) |
| `emission_prob` | model probability that the point lies on this road (`predict_proba[:, 1]`) |

Probabilities don't need to sum to 1 per point. Keep 3-8 candidates per point
(e.g. all segments within ~50-100 m).

**Output**: `matched.csv` (`trip_id,point_idx,road_id`) and
`matched_routes.csv` (`trip_id,route` with space-separated road ids).

## Demo on mocked data

```bash
python demo_viterbi.py            # 4x4 toy grid, 25 m GPS noise
python -m pytest -q tests
python -m hmm.run --graph examples/mock_roads.csv --candidates examples/mock_candidates.csv --out matched.csv
```

On 50 random synthetic trips at 25 m noise: per-point argmax 73.5% correct,
Viterbi 85.2%, and the reconstructed route matches the true route exactly on
the demo trip. Remaining errors are points right at intersections.

## Tuning knobs (`HMMConfig`)

`beta` (50 m): tolerance for route vs. straight-line mismatch. `turn_weight` (3):
penalty for turns/U-turns. `backtrack_tol` (30 m): backward jitter on the same
road that is ignored. `emission_weight` (1): trust in the model vs. the graph.

## FastAPI integration (`api/pipeline.py`)

`hmm.api_adapter.viterbi_for_api(net, track, layers)` has the same signature and
return value as `Pipeline.viterbi` (chosen candidate index per layer), so the API
stage can simply be:

```python
from hmm.api_adapter import viterbi_for_api

def viterbi(self, net, track, layers):
    return viterbi_for_api(net, track, layers)
```

Directed arcs are treated as one-way segments (a two-way street = two arcs).
