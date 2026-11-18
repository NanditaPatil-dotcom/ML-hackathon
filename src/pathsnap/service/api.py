"""
PathSnap FastAPI Server (Laptop 1):
Hosts the trained model, HMM Viterbi decoder, and reference ground truth data.
Exposes evaluation endpoints and logs requests cleanly in the terminal for the evaluator.
"""

import os
import time
import logging
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathsnap.decoder.pipeline import reconstruct_trip_with_scorer
from pathsnap.models.scorer import CandidateScorer

# Configure formatted terminal logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PathSnap API] %(message)s"
)
logger = logging.getLogger("pathsnap")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# Override with environment variables when the data or model live elsewhere.
DATA_DIR = os.environ.get("PATHSNAP_DATA_DIR", os.path.join(REPO_ROOT, "Student folder"))
MODEL_PATH = os.environ.get(
    "PATHSNAP_MODEL", os.path.join(REPO_ROOT, "artifacts", "models", "catboost_best.joblib")
)
# Partitions searched for a trip id (/trips shows evaluator_test if present, else student_test).
PARTITIONS = ["student_test", "evaluator_test", "train"]

app = FastAPI(title="PathSnap Distributed Map-Matching API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
MODEL = None

def _load_model():
    global MODEL
    if MODEL is None:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found at {MODEL_PATH}!")
            return None
        logger.info(f"Loading CandidateScorer from: {MODEL_PATH}...")
        MODEL = CandidateScorer.load(MODEL_PATH)
        logger.info("CandidateScorer ready.")
    return MODEL


def _find_trip_dir(trip_id):
    for part in PARTITIONS:
        d = os.path.join(DATA_DIR, part)
        if os.path.isdir(os.path.join(d, trip_id)):
            return d
    return None


@app.on_event("startup")
def load_model_on_startup():
    _load_model()

@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "PathSnap Distributed Map-Matching API",
        "model_loaded": MODEL is not None
    }

@app.get("/trips")
def list_available_trips():
    """Lists the evaluation trips available on the server."""
    # The reserved evaluation trips are shown when present; otherwise the
    # student_test trips (development).
    for part in ("evaluator_test", "student_test"):
        test_dir = os.path.join(DATA_DIR, part)
        if os.path.isdir(test_dir):
            trips = sorted(d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d)))
            if trips:
                return {"trips": trips}
    return {"trips": []}

@app.post("/evaluate/{trip_id}")
def evaluate_trip(trip_id: str):
    t_start = time.time()
    logger.info(f">>> RECEIVED INFERENCE EVALUATION REQUEST FOR TRIP ID: {trip_id} <<<")

    scorer = _load_model()
    if scorer is None:
        raise HTTPException(status_code=500, detail="Model artifact missing")

    partition_dir = _find_trip_dir(trip_id) if trip_id.isalnum() else None
    if partition_dir is None:
        logger.error(f"Trip ID '{trip_id}' not found under {DATA_DIR}")
        raise HTTPException(status_code=404, detail=f"Trip {trip_id} not found")

    logger.info(f"Processing trip data from: {partition_dir}...")
    res = reconstruct_trip_with_scorer(partition_dir, trip_id, scorer)
    if res is None:
        raise HTTPException(status_code=500, detail="Failed to process trip data")
    m = res["metrics"]
    # Keys the Streamlit UI reads.
    m["clf_top1_acc"] = m.get("clf_acc", 0.0)
    m["hmm_top1_acc"] = m.get("hmm_acc", 0.0)

    duration = round(time.time() - t_start, 3)
    logger.info(f"Inference complete in {duration}s!")
    logger.info(f"Evaluation Metrics for Trip {trip_id}:")
    logger.info(f"  Classifier Top-1 Accuracy: {res['metrics'].get('clf_top1_acc', 0):.2f}%")
    logger.info(f"  HMM/Viterbi Top-1 Accuracy: {res['metrics'].get('hmm_top1_acc', 0):.2f}%")
    logger.info(f"  Classifier Route Continuity: {res['metrics']['clf_continuity']:.2f}%")
    logger.info(f"  HMM/Viterbi Route Continuity: {res['metrics']['hmm_continuity']:.2f}%")
    logger.info("Returning evaluation package to UI client...\n")
    
    # Format coordinate segments for visualization
    track = res["track"]
    nodes = res["nodes"]
    arcs = res["arcs"]
    hmm_path = []
    for t in sorted(res["hmm_path"]):  # one arc per GPS point -> drop consecutive repeats
        if not hmm_path or hmm_path[-1] != res["hmm_path"][t]:
            hmm_path.append(res["hmm_path"][t])
    
    # Subsampled GPS track coordinates for map rendering (lat, lon)
    xy = track[["lon", "lat"]].to_numpy()
    gps_coords = [[float(p[1]), float(p[0])] for p in xy[::max(1, len(xy)//400)]]
    
    # Ground truth stays on this server: it is only used above to score the
    # prediction and is never sent to the UI.

    # Predicted route segments (lat, lon pairs)
    pred_segments = []
    for arc_id in hmm_path:
        if arc_id < len(arcs):
            u, v = arcs[arc_id]
            if u < len(nodes) and v < len(nodes):
                pred_segments.append([
                    [float(nodes[u, 1]), float(nodes[u, 0])],
                    [float(nodes[v, 1]), float(nodes[v, 0])]
                ])
                
    return {
        "status": "success",
        "trip_id": trip_id,
        "processing_time_sec": duration,
        "num_gps_observations": len(track),
        "metrics": res["metrics"],
        "gps_coords": gps_coords,
        "predicted_segments": pred_segments
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
