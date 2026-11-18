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
