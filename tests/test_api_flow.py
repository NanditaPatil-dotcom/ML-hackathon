"""
Integration test for PathSnap FastAPI and Evaluator flow.
Tests health check, trips enumeration, and inference endpoint on evaluator_test set.
"""
from fastapi.testclient import TestClient
from pathsnap.service.api import app, PARTITIONS, DATA_DIR
import os

client = TestClient(app)

def test_api_health():
    res = client.get("/")
    assert res.status_code == 200
    assert res.json()["status"] == "online"

def test_api_list_trips():
    res = client.get("/trips")
    assert res.status_code == 200
    trips = res.json().get("trips", [])
    assert len(trips) == 10
    assert "00000015" in trips

def test_api_evaluate_trip_security():
    res = client.post("/evaluate/00000015")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    # Ground truth route must NEVER be exposed to UI/client
    assert "ground_truth_segments" not in data
    # Predicted segments must be present
    assert "predicted_segments" in data
    assert len(data["predicted_segments"]) > 0
    # Metrics must be computed server-side
    assert "hmm_continuity" in data["metrics"]
    assert "hmm_top1_acc" in data["metrics"]
    assert data["metrics"]["hmm_continuity"] >= 80.0

def test_no_train_fallback():
    # Attempting to query a training trip should return 404 because train is excluded from PARTITIONS
    res = client.post("/evaluate/00000001")
    assert res.status_code == 404

if __name__ == "__main__":
    test_api_health()
    print("test_api_health: PASS")
    test_api_list_trips()
    print("test_api_list_trips: PASS")
    test_api_evaluate_trip_security()
    print("test_api_evaluate_trip_security: PASS")
    test_no_train_fallback()
    print("test_no_train_fallback: PASS")
    print("ALL API FLOW TESTS PASSED!")
