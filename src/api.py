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
from src.hmm_viterbi import reconstruct_route_for_trip

# Configure formatted terminal logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PathSnap API] %(message)s"
)
logger = logging.getLogger("pathsnap")

DATA_DIR = "/home/wingsfry/Projects/ML hackathon/Student folder"
ARTIFACTS_DIR = "/home/wingsfry/Projects/ML hackathon/artifacts"

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

@app.on_event("startup")
def load_model_on_startup():
    global MODEL
    model_path = os.path.join(ARTIFACTS_DIR, "best_model.joblib")
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}!")
        return
    logger.info(f"Loading trained model artifact from: {model_path}...")
    MODEL = joblib.load(model_path)
    logger.info("PathSnap ML Model successfully loaded and ready for evaluation requests.")

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
    test_dir = os.path.join(DATA_DIR, "student_test")
    if not os.path.exists(test_dir):
        return {"trips": []}
    trips = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    return {"trips": trips}

@app.post("/evaluate/{trip_id}")
def evaluate_trip(trip_id: str):
    global MODEL
    t_start = time.time()
    logger.info(f">>> RECEIVED INFERENCE EVALUATION REQUEST FOR TRIP ID: {trip_id} <<<")
    
    if MODEL is None:
        model_path = os.path.join(ARTIFACTS_DIR, "best_model.joblib")
        if os.path.exists(model_path):
            MODEL = joblib.load(model_path)
        else:
            logger.error("No model artifact loaded!")
            raise HTTPException(status_code=500, detail="Model artifact missing")
            
    # Locate trip directory
    trip_dir = os.path.join(DATA_DIR, "student_test", trip_id)
    if not os.path.exists(trip_dir):
        trip_dir = os.path.join(DATA_DIR, "train", trip_id)
        if not os.path.exists(trip_dir):
            logger.error(f"Trip ID '{trip_id}' not found in student_test or train datasets!")
            raise HTTPException(status_code=404, detail=f"Trip {trip_id} not found")
            
    logger.info(f"Processing trip data from: {trip_dir}...")
    res = reconstruct_route_for_trip(trip_dir, trip_id, MODEL)
    if res is None:
        raise HTTPException(status_code=500, detail="Failed to process trip data")
        
    duration = round(time.time() - t_start, 3)
    logger.info(f"Inference complete in {duration}s!")
    logger.info(f"Evaluation Metrics for Trip {trip_id}:")
    logger.info(f"  Classifier Top-1 Accuracy: {res['metrics'].get('clf_top1_acc', 'N/A')}%")
    logger.info(f"  HMM/Viterbi Top-1 Accuracy: {res['metrics'].get('hmm_top1_acc', 'N/A')}%")
    logger.info(f"  Classifier Route Continuity: {res['metrics'].get('clf_continuity', 'N/A'):.2f}%")
    logger.info(f"  HMM/Viterbi Route Continuity: {res['metrics'].get('hmm_continuity', 'N/A'):.2f}%")
    logger.info("Returning evaluation package to UI client...\n")
    
    # Format coordinate segments for visualization
    track = res["track"]
    nodes = res["nodes"]
    arcs = res["arcs"]
    route = res["route"]
    hmm_path = res["hmm_path"]
    
    # Subsampled GPS track coordinates for map rendering (lat, lon)
    gps_coords = [[float(p[1]), float(p[0])] for p in track[::max(1, len(track)//400)]]
    
    # Ground truth route segments (lat, lon pairs)
    gt_segments = []
    if route is not None:
        for arc_id in route:
            if arc_id < len(arcs):
                u, v = arcs[arc_id]
                if u < len(nodes) and v < len(nodes):
                    gt_segments.append([
                        [float(nodes[u, 1]), float(nodes[u, 0])],
                        [float(nodes[v, 1]), float(nodes[v, 0])]
                    ])
                    
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
        "ground_truth_segments": gt_segments,
        "predicted_segments": pred_segments
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
