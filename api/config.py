"""Runtime configuration, all overridable with environment variables.

Set these on Laptop 1 before starting the server, e.g. (PowerShell):
    $env:PATHSNAP_MODEL_PATH = "models\\final_model.joblib"
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

HOST = os.getenv("PATHSNAP_HOST", "0.0.0.0")  # 0.0.0.0 = reachable from other laptops
PORT = int(os.getenv("PATHSNAP_PORT", "8000"))

# Trained road-candidate classifier (joblib/pickle with predict_proba). Optional:
# when missing, a distance+heading heuristic scores candidates instead.
# Default: first that exists of models/final_model.joblib, artifacts/models/catboost_best.joblib.
_default_model = next((p for p in (ROOT / "models" / "final_model.joblib",
                                   ROOT / "artifacts" / "models" / "catboost_best.joblib") if p.exists()),
                      ROOT / "artifacts" / "models" / "catboost_best.joblib")
MODEL_PATH = Path(os.getenv("PATHSNAP_MODEL_PATH", _default_model))

# Raw dataset root. Every <trip>.track found below it (recursively) is a trip,
# with sibling <trip>.nodes / <trip>.arcs / <trip>.route files.
DATA_DIR = Path(os.getenv("PATHSNAP_DATA_DIR", ROOT / "data" / "raw"))

# Which trips the UI offers (the 10 evaluator-reserved trips). Either:
#   PATHSNAP_EVAL_TRIPS="tripA,tripB,..."  (explicit IDs), or
#   PATHSNAP_EVAL_SUBDIR="evaluator"       (trips under a folder whose path contains this text).
# If neither matches anything, every trip under DATA_DIR is offered.
EVAL_TRIPS = [t.strip() for t in os.getenv("PATHSNAP_EVAL_TRIPS", "").split(",") if t.strip()]
EVAL_SUBDIR = os.getenv("PATHSNAP_EVAL_SUBDIR", "eval")

# Candidate generation / HMM parameters (good targets for tuning).
CANDIDATE_RADIUS_M = float(os.getenv("PATHSNAP_CANDIDATE_RADIUS_M", "50"))
MAX_CANDIDATES = int(os.getenv("PATHSNAP_MAX_CANDIDATES", "8"))
GPS_SIGMA_M = float(os.getenv("PATHSNAP_GPS_SIGMA_M", "10"))
TRANSITION_BETA_M = float(os.getenv("PATHSNAP_TRANSITION_BETA_M", "20"))
UNCONNECTED_PENALTY = float(os.getenv("PATHSNAP_UNCONNECTED_PENALTY", "3.0"))  # log-prob cost for jumping between non-adjacent arcs
MAX_SPEED_MPS = float(os.getenv("PATHSNAP_MAX_SPEED_MPS", "60"))  # ~216 km/h; faster jumps are dropped as noise
