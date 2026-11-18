# PathSnap — Data Processing & Feature Engineering

This module covers the **data preprocessing + feature engineering** stage of the PathSnap
pipeline: turning raw per-trip GPS/road-network files into clean, feature-rich tables that
the **model-training** team and the **HMM/Viterbi** team can build on directly.

## Raw data format (per trip, no headers, whitespace-separated)

| File | Columns | Notes |
|---|---|---|
| `<trip>.track` | `lon, lat, timestamp_sec` | raw GPS observations, nominally 1 Hz |
| `<trip>.nodes` | `lon, lat` | road-network nodes; 0-based row index = `node_id` |
| `<trip>.arcs` | `from_node_id, to_node_id` | road segments; 0-based row index = `arc_id` |
| `<trip>.route` | `arc_id` (one per row) | chronological ground-truth arc sequence (train/student-test only) |

Dataset partitions: **80 train / 10 student-test / 10 evaluator-test** (evaluator-test is held
out — never touched by this pipeline for fitting anything, only usable at true inference time).

## Pipeline stages (`pipeline.py`)

1. **Load** — discover all trips in a partition directory and load their 4 raw files
   (`io_utils.py`).
2. **Clean** (`clean.py`) —
   - drop null/duplicate/out-of-range GPS rows,
   - enforce monotonic timestamps,
   - filter points implying impossible speed (> `MAX_SPEED_MPS`) or acceleration
     (> `MAX_ACCEL_MPS2`) jumps between consecutive points (geodesic distance / dt).
3. **Road graph** (`road_graph.py`) — build a directed graph from `.nodes`/`.arcs` with a
   KD-tree spatial index over segment midpoints for fast nearest-road lookups, and expose
   per-arc metadata (length, bearing, a synthetic speed-limit proxy from typical traversal
   speeds, and node adjacency for the HMM's transition graph).
4. **Feature engineering** (`features.py`) — for every cleaned GPS point:
   - generate **candidate arcs** within `CANDIDATE_RADIUS_M` of the point,
   - compute **distance-to-segment** (perpendicular geodesic distance to the arc polyline),
   - compute **bearing alignment** (difference between the vehicle's instantaneous bearing
     and the arc's bearing),
   - compute the arc's **speed-limit proxy**,
   - compute **inter-point transition distance** (geodesic distance between consecutive raw
     GPS points, and graph distance between consecutive candidate arcs) — these feed the
     HMM's emission/transition probabilities downstream.
5. **Split** (`split.py`) — train/test split **by trip id**, never by point, matching the
   80/10 partitioning already provided in the raw data (if the raw folders are already split
   into partitions, this stage just validates that; if a team member needs a further
   train/val split within the 80 training trips, it does that split by trip too).

## Output format

Run from the repository root with `python scripts/preprocess.py --data-root <path> --out-dir results/preprocessing`. The command writes, per partition:

- `<out-dir>/<partition>_candidates.parquet` — one row per **(trip_id, point_idx, candidate_arc_id)**:

  | column | meaning |
  |---|---|
  | `trip_id` | trip identifier |
  | `point_idx` | 0-based index of the GPS point within its cleaned track |
  | `timestamp` | seconds |
  | `lon`, `lat` | cleaned GPS coordinates |
  | `candidate_arc_id` | candidate road segment id |
  | `dist_to_segment_m` | perpendicular distance, metres |
  | `bearing_alignment_deg` | absolute bearing difference, degrees |
  | `speed_limit_proxy_mps` | arc speed-limit proxy |
  | `point_speed_mps` | observed speed from previous point |
  | `is_ground_truth_arc` | 1 if this candidate matches the `.route` arc at this point (train/student-test only) |

- `<out-dir>/<partition>_transitions.parquet` — one row per consecutive GPS point pair per trip:
  `trip_id, point_idx, dt_sec, point_transition_dist_m` (feeds HMM transition probabilities
  together with the road graph's arc-to-arc adjacency, which the HMM/Viterbi team should load
  from `road_graph.RoadGraph.arc_adjacency(trip_id)`).

- `<out-dir>/<partition>_trips.csv` — one row per trip: trip-level summary stats
  (num points, duration, candidate miss rate, etc.) for the EDA/report.

Model-training team: treat `*_candidates.parquet` as your supervised table — group by
`(trip_id, point_idx)`, and `is_ground_truth_arc` is your label for the candidate classifier.

HMM/Viterbi team: candidates + their scores are your emission inputs; `road_graph.RoadGraph`
gives you segment connectivity for transitions.
