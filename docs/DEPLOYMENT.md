# PathSnap deployment: two-laptop setup

| Laptop | Runs | Holds |
|---|---|---|
| **Laptop 1** (Model + API) | `run_api.bat` → FastAPI on port **8000** | trained model, HMM/Viterbi, road network + trip files, reference routes (ground truth stays here) |
| **Laptop 2** (Evaluator UI) | `run_ui.bat` → static UI on port **5500** | only the `ui/` folder (+ `scripts/client.py` for the CMD demo) |

Flow: UI lists the 10 evaluation trips (`GET /trips`) → evaluator clicks one → UI sends `POST /predict {"trip_id": ...}` → API runs clean → candidates → classifier → HMM/Viterbi, scores it against its own reference route, returns route + accuracy → UI draws it. The UI never sees ground truth.

## API contract

Base URL: `http://<LAPTOP1_IP>:8000` (Swagger docs at `/docs`).

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | — | `status`, `trips_loaded`, `evaluation_trips`, which stages are real vs stub |
| GET | `/trips` | — | `{"count": 10, "trips": [{"trip_id": "...", "num_observations": 812}, ...]}` |
| POST | `/predict` | `{"trip_id": "..."}` | see below |
| POST | `/reload` | — | re-reads model + data from disk without restarting |

`POST /predict` response:

```json
{
  "trip_id": "demo_01",
  "num_observations": 140,
  "num_observations_used": 140,
  "num_outliers_removed": 0,
  "predicted_arc_sequence": [2, 32, 60, 66, 94],
  "per_observation_arcs": [2, 2, 2, 32, null, 32],
  "predicted_route_coords": [[77.53, 12.93], [77.531, 12.93]],
  "gps_points": [[77.5301, 12.9302], [77.5302, 12.9301]],
  "hmm_accuracy": 0.9857,
  "classifier_only_accuracy": 0.9143,
  "metrics": {
    "hmm":             {"observation_accuracy": 0.9857, "arc_precision": 0.875, "arc_recall": 1.0, "arc_f1": 0.9333, "sequence_similarity": 0.9333},
    "classifier_only": {"observation_accuracy": 0.9143, "arc_precision": 0.56,  "arc_recall": 1.0, "arc_f1": 0.7179, "sequence_similarity": 0.6087}
  },
  "processing_time_ms": 29.2,
  "pipeline": {"scoring": "model RandomForestClassifier (final_model.joblib)", "...": "..."}
}
```

Coordinates are `[lon, lat]` (same order as the raw files). `metrics` is `null` if a trip has no `.route` file. Errors: `404` unknown trip, `422` bad body / empty track.

Metric definitions (all computed server-side against `<trip>.route`):
- `observation_accuracy`: share of matched GPS points whose arc is on the reference route.
- `arc_precision / arc_recall / arc_f1`: over the set of arcs in predicted vs reference route.
- `sequence_similarity`: `2·LCS / (len(pred)+len(ref))` over the collapsed arc sequences.

## Laptop 1 setup (once, before the demo)

1. Clone the repo, then `pip install -r requirements-api.txt`.
2. Put the raw trip files under `data/raw/` (any sub-folders are fine; every `<trip>.track` with its `.nodes/.arcs/.route` siblings becomes a trip). Put the 10 evaluator trips in a folder whose name contains `eval` (e.g. `data/raw/evaluator_test/`), **or** set `PATHSNAP_EVAL_TRIPS=id1,id2,...`.
3. Model and pipeline code: the API picks up each teammate's piece automatically when it is in the repo, and `/health` shows which are live:
   - cleaning + candidates + features: `data_processing/` (`clean_track`, `build_candidates_table`), the same code that built the training set;
   - scorer: `models/final_model.joblib` if present, else `artifacts/models/catboost_best.joblib` (the `CandidateScorer` from `src/pathsnap/models/scorer.py`, fed `dist_to_segment_m, bearing_alignment_deg, speed_limit_proxy_mps, point_speed_mps, arc_length_m, candidate_rank`);
   - decoder: `hmm/api_adapter.py`.
   Any piece that is missing or fails falls back to a builtin version so the demo never breaks.
   Install every team requirement on Laptop 1: `pip install -r requirements-api.txt -r requirements.txt -r data_processing/requirements.txt` (skip files that don't exist yet).
4. Start: `run_api.bat` (or `python -m api.main`). You should see `Uvicorn running on http://0.0.0.0:8000`.
5. Check locally: open `http://localhost:8000/health` in a browser.

### Allow the port through Windows Firewall (Laptop 1)

The first time Python listens on a port, Windows pops up "Allow access" — tick **Private networks** (and Public if the venue Wi-Fi is classed as public), then **Allow**. If you missed it, run this in an **Administrator** PowerShell:

```powershell
New-NetFirewallRule -DisplayName "PathSnap API 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Any
```

Remove it after the hackathon: `Remove-NetFirewallRule -DisplayName "PathSnap API 8000"`.

macOS: System Settings → Network → Firewall → allow incoming for Python. Linux: `sudo ufw allow 8000/tcp`.

### Find Laptop 1's IP

- Windows: `ipconfig` → **Wireless LAN adapter Wi-Fi → IPv4 Address** (e.g. `192.168.1.23`). Ignore VirtualBox/WSL/vEthernet adapters.
- macOS: `ipconfig getifaddr en0`. Linux: `hostname -I`.

The IP can change if Laptop 1 reconnects, so re-check it right before the demo.

## Laptop 2 setup

1. Copy the repo (only `ui/` and `scripts/client.py` are needed). Python is only needed to serve the page; `pip install requests` for the CLI client.
2. Test the network first: `ping <LAPTOP1_IP>` then open `http://<LAPTOP1_IP>:8000/health` in a browser. If that works, everything else will.
3. Start the UI: `run_ui.bat` (or `cd ui && python -m http.server 5500`).
4. Open `http://localhost:5500/?api=http://<LAPTOP1_IP>:8000`. The API box is remembered, and you can edit it and press **Connect**.
5. CMD demo (the report needs screenshots of API responses in CMD):
   `python scripts/client.py --api http://<LAPTOP1_IP>:8000 --all`
   Laptop 1's console also logs one `PREDICT trip=... hmm_acc=...` line per request; show the evaluator that window.

Leaflet is bundled in `ui/vendor/`, so the UI works without internet; only the OpenStreetMap background tiles need internet (the route and GPS points still draw on a grey background without it).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `/health` works on Laptop 1 but not from Laptop 2 | Firewall rule (above); both laptops on the same Wi-Fi/SSID; venue Wi-Fi may isolate clients ("AP/client isolation"). If so, use a **phone hotspot** with both laptops joined to it. |
| Works by IP from Laptop 2 browser but UI says "Cannot reach" | API box must include `http://` and `:8000`; open the browser dev tools console for the exact error. |
| `Address already in use` | Another server on 8000: close it or `set PATHSNAP_PORT=8001` and use `:8001` in the UI. |
| `/trips` returns 0 trips | `PATHSNAP_DATA_DIR` wrong or files not unzipped; `/health` shows the directory used. |
| Trips list shows train trips too | No folder name contains `eval`; set `PATHSNAP_EVAL_TRIPS` or `PATHSNAP_EVAL_SUBDIR`. |
| `scoring: STUB` in `/health` | `models/final_model.joblib` missing or failed to load (see console); call `POST /reload` after copying it in. |

## Demo-day checklist

- [ ] Both laptops charged, on the same network (have a phone hotspot as backup).
- [ ] Laptop 1: `git pull`, model in `models/`, data in `data/raw/`, `run_api.bat` running, `/health` shows model loaded (not STUB) and 10 evaluation trips.
- [ ] Laptop 1 IP noted; firewall rule in place.
- [ ] Laptop 2: `/health` opens in browser via IP, `run_ui.bat` running, UI shows 10 trips.
- [ ] Ran `scripts/client.py --all` once: all 10 trips return 200.
- [ ] Laptop 1 console window visible to show request logs.
