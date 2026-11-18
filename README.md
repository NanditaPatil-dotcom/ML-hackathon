# ML-hackathon: PathSnap

GPS map-matching: noisy GPS trip → candidate road arcs → classifier → HMM/Viterbi → most likely route, served by a FastAPI backend (Laptop 1) to an evaluator UI (Laptop 2).

## Repository layout

| Folder | Contents |
|---|---|
| `api/` | FastAPI server: `main.py` (endpoints), `pipeline.py` (clean → candidates → scoring → Viterbi), `data.py` (raw file loading), `metrics.py` (server-side accuracy), `config.py` (env-var settings) |
| `ui/` | Evaluator interface (static HTML + bundled Leaflet) |
| `scripts/` | `client.py` CLI evaluator, `make_demo_data.py` synthetic trips for testing |
| `docs/` | `DEPLOYMENT.md`: API contract and two-laptop setup |
| `models/` | final trained model (`final_model.joblib`) |
| `data/raw/` | raw trip files (`.track/.nodes/.arcs/.route`) |

## Run

```bash
pip install -r requirements-api.txt

# Laptop 1 (API)
python -m api.main                 # or run_api.bat; serves http://0.0.0.0:8000, docs at /docs

# Laptop 2 (UI)
cd ui && python -m http.server 5500   # or run_ui.bat
# open http://localhost:5500/?api=http://<LAPTOP1_IP>:8000
python scripts/client.py --api http://<LAPTOP1_IP>:8000 --all
```

Try it without the real data: `python scripts/make_demo_data.py data/demo` then `PATHSNAP_DATA_DIR=data/demo python -m api.main`.

Full API contract, firewall/IP steps and troubleshooting: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

### Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `PATHSNAP_DATA_DIR` | `data/raw` | root of raw trip files (searched recursively) |
| `PATHSNAP_MODEL_PATH` | `models/final_model.joblib` | trained classifier; heuristic fallback if missing |
| `PATHSNAP_EVAL_TRIPS` | — | comma-separated trip IDs to list in the UI |
| `PATHSNAP_EVAL_SUBDIR` | `eval` | otherwise, list trips in folders whose name contains this |
| `PATHSNAP_PORT` / `PATHSNAP_HOST` | `8000` / `0.0.0.0` | bind address |
| `PATHSNAP_CANDIDATE_RADIUS_M`, `PATHSNAP_MAX_CANDIDATES`, `PATHSNAP_GPS_SIGMA_M`, `PATHSNAP_TRANSITION_BETA_M`, `PATHSNAP_UNCONNECTED_PENALTY` | 50, 8, 10, 20, 3.0 | candidate + HMM parameters |

## Candidate-segment model training (CatBoost + RBF SVM)

Trains and compares model families that score each GPS point's candidate road
segments, for the HMM/Viterbi step downstream. This script covers **CatBoost**
and **RBF SVM**; teammates are covering LightGBM, Extra Trees and Random Forest
in parallel against the same interface, so the full 5-model comparison the
task spec requires comes together from both.

Real preprocessing isn't ready yet, so this is built and tested against a
mocked dataset with the same column names the real pipeline will produce:
`distance_to_segment, bearing_diff, speed_limit, transition_distance,
candidate_road_id, label` (plus `point_id`/`trip_id` for grouped splits).

```bash
pip install -r requirements.txt

# generate the mock dataset
python src/generate_mock_data.py

# train + compare, write outputs/model_comparison.{csv,md} and outputs/best_model.joblib
python src/train_compare.py --data data/mock/candidates_mock.csv --out-dir outputs

# once real preprocessed data lands, just point at it (same column names):
python src/train_compare.py --data path/to/real_preprocessed.csv --out-dir outputs
```

### Results on the mock dataset

| Model | CV log_loss | CV top1_point_acc | test predict_ms |
|---|---|---|---|
| RBF_SVM | 0.2223 ± 0.0125 | 0.858 | 60.9 |
| CatBoost | 0.2223 ± 0.0149 | 0.859 | 1.5 |

Statistically tied on log-loss; **CatBoost** wins on inference speed (needed
since the HMM scores every candidate of every GPS point) and is saved as
`outputs/best_model.joblib`.

### Using the trained model (for the HMM step)

```python
from candidate_scorer import CandidateScorer

scorer = CandidateScorer.load("outputs/best_model.joblib")
df["p_match"] = scorer.predict_proba(df)        # P(candidate is the true road), in [0,1]
df["emission"] = scorer.emission_log_prob(df)   # log-prob, normalized per point_id
```
