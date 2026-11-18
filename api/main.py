"""PathSnap FastAPI server (Laptop 1).

    python -m api.main            # binds 0.0.0.0:8000 (see api/config.py for env vars)

Endpoints
    GET  /health          liveness + which pipeline stages are real vs stubbed
    GET  /trips           the evaluator trip IDs the UI should list
    POST /predict         {"trip_id": "..."} -> predicted route + server-side accuracy
    POST /reload          re-read model/data from disk without restarting
    GET  /docs            Swagger UI
"""
import logging
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .data import TripIndex
from .metrics import route_metrics
from .pipeline import Pipeline
from .schemas import PredictRequest, PredictResponse, TripInfo, TripsResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("pathsnap.api")

app = FastAPI(title="PathSnap Map-Matching API", version="1.0.0",
              description="Trip ID -> cleaned GPS -> candidate arcs -> classifier -> HMM/Viterbi -> route + accuracy.")
# The UI is served from Laptop 2 (different origin), so allow cross-origin browser calls.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

pipeline = Pipeline()
trips = TripIndex()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    resp = await call_next(request)
    log.info("%s %s %s from %s in %.0f ms", request.method, request.url.path, resp.status_code,
             request.client.host if request.client else "?", (time.perf_counter() - t0) * 1000)
    return resp


@app.get("/")
def root():
    return {"service": "PathSnap", "status": "ok", "docs": "/docs", "endpoints": ["/health", "/trips", "/predict"]}


@app.get("/health")
def health():
    return {"status": "ok", "trips_loaded": len(trips.trips), "evaluation_trips": len(trips.evaluation_trips()),
            "data_dir": str(config.DATA_DIR), "pipeline": pipeline.status()}


@app.get("/trips", response_model=TripsResponse)
def list_trips():
    out = []
    for tid in trips.evaluation_trips():
        try:
            n = sum(1 for line in trips.get(tid).track.open() if line.strip())
        except OSError:
            n = 0
        out.append(TripInfo(trip_id=tid, num_observations=n))
    return TripsResponse(count=len(out), trips=out)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    tid = str(req.trip_id)
    tf = trips.get(tid)
    if tf is None:
        raise HTTPException(status_code=404, detail=f"Unknown trip_id '{tid}'. GET /trips for valid IDs.")
    t0 = time.perf_counter()
    track, nodes, arcs = tf.load_track(), tf.load_nodes(), tf.load_arcs()
    if len(track) == 0:
        raise HTTPException(status_code=422, detail=f"Trip '{tid}' has no GPS observations")
    res = pipeline.run(track, nodes, arcs)

    # Ground truth is read only here, after prediction, and only metrics leave the server.
    ref = tf.load_reference_route()
    metrics = None
    if ref:
        metrics = {"hmm": route_metrics(res["route_hmm"], res["per_observation_arcs"], ref),
                   "classifier_only": route_metrics(res["route_clf"], res["classifier_only_arcs"], ref)}

    out = PredictResponse(
        trip_id=tid,
        num_observations=len(track),
        num_observations_used=int(res["kept_mask"].sum()),
        num_outliers_removed=int((~res["kept_mask"]).sum()),
        predicted_arc_sequence=res["route_hmm"],
        per_observation_arcs=res["per_observation_arcs"],
        predicted_route_coords=res["network"].arc_coords(res["route_hmm"]),
        gps_points=[[float(r[0]), float(r[1])] for r in track],
        hmm_accuracy=metrics["hmm"]["observation_accuracy"] if metrics else None,
        classifier_only_accuracy=metrics["classifier_only"]["observation_accuracy"] if metrics else None,
        metrics=metrics,
        processing_time_ms=round((time.perf_counter() - t0) * 1000, 1),
        pipeline=pipeline.status(),
    )
    log.info("PREDICT trip=%s obs=%d used=%d arcs=%d hmm_acc=%s clf_acc=%s time=%.0fms", tid, out.num_observations,
             out.num_observations_used, len(out.predicted_arc_sequence), out.hmm_accuracy,
             out.classifier_only_accuracy, out.processing_time_ms)
    return out


@app.post("/reload")
def reload_all():
    global pipeline, trips
    pipeline, trips = Pipeline(), TripIndex()
    return health()


if __name__ == "__main__":
    import uvicorn

    log.info("Starting PathSnap API on http://%s:%d  (docs at /docs)", config.HOST, config.PORT)
    uvicorn.run(app, host=config.HOST, port=config.PORT)
