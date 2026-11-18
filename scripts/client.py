"""Command-line evaluator client (Laptop 2). Prints API responses in the CMD window.

    python scripts/client.py --api http://<LAPTOP1_IP>:8000            # list trips + health
    python scripts/client.py --api http://<LAPTOP1_IP>:8000 --trip 12  # run one trip
    python scripts/client.py --api http://<LAPTOP1_IP>:8000 --all      # run every evaluation trip
"""
import argparse
import json
import sys

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--trip")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--full", action="store_true", help="print the whole JSON response")
    a = ap.parse_args()
    base = a.api.rstrip("/")
    try:
        print("health:", json.dumps(requests.get(f"{base}/health", timeout=5).json(), indent=2))
    except requests.RequestException as e:
        sys.exit(f"Cannot reach {base}: {e}\nCheck Laptop 1's IP, that the server is running, and the firewall rule.")
    trips = requests.get(f"{base}/trips", timeout=10).json()["trips"]
    print(f"{len(trips)} evaluation trips:", ", ".join(t["trip_id"] for t in trips))
    targets = [t["trip_id"] for t in trips] if a.all else ([a.trip] if a.trip else [])
    for tid in targets:
        r = requests.post(f"{base}/predict", json={"trip_id": tid}, timeout=120)
        d = r.json()
        if not r.ok:
            print(f"[{tid}] ERROR {r.status_code}: {d}")
            continue
        if a.full:
            print(json.dumps(d, indent=2))
        else:
            print(f"[{tid}] HTTP {r.status_code}  obs={d['num_observations']}  arcs={len(d['predicted_arc_sequence'])}  "
                  f"hmm_acc={d['hmm_accuracy']}  clf_acc={d['classifier_only_accuracy']}  {d['processing_time_ms']} ms")
            print("   metrics:", json.dumps(d["metrics"]))


if __name__ == "__main__":
    main()
