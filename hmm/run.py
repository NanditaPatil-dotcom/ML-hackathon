"""Batch map-matching from files produced by preprocessing + the model.

    python -m hmm.run --graph roads.csv --candidates candidates.csv --out matched.csv

roads.csv       road_id,u,v,u_lat,u_lon,v_lat,v_lon[,oneway][,length_m][,geometry]
                (geometry optional: "lat lon;lat lon;..." polyline incl. endpoints)
                -- or a roads.json: {"nodes": {id: [lat, lon]}, "segments": [{road_id,u,v,...}]}
candidates.csv  trip_id,point_idx,lat,lon,road_id,emission_prob   (one row per candidate)
matched.csv     trip_id,point_idx,road_id            + matched_routes.csv: trip_id,route
"""
import argparse
import csv
from collections import defaultdict

from .graph import RoadGraph
from .matcher import HMMConfig, match_trip

TRUE = {"1", "true", "yes", "y", "t"}


def load_graph(path):
    if path.endswith(".json"):
        return RoadGraph.from_json(path)
    nodes, segs = {}, []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            nodes[r["u"]] = (float(r["u_lat"]), float(r["u_lon"]))
            nodes[r["v"]] = (float(r["v_lat"]), float(r["v_lon"]))
            s = {"road_id": r["road_id"], "u": r["u"], "v": r["v"],
                 "oneway": str(r.get("oneway", "")).strip().lower() in TRUE}
            if r.get("length_m"):
                s["length"] = float(r["length_m"])
            if r.get("geometry"):
                s["geometry"] = [tuple(map(float, p.split())) for p in r["geometry"].split(";")]
            segs.append(s)
    return RoadGraph(nodes, segs)


def load_candidates(path):
    """-> {trip_id: [ {lat, lon, candidates:[(road_id, p)], point_idx}, ... ] } sorted by point_idx."""
    trips = defaultdict(dict)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            idx = int(r["point_idx"])
            pt = trips[r["trip_id"]].setdefault(
                idx, {"lat": float(r["lat"]), "lon": float(r["lon"]), "candidates": [], "point_idx": idx})
            pt["candidates"].append((r["road_id"], float(r["emission_prob"])))
    return {t: [pts[i] for i in sorted(pts)] for t, pts in trips.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", default="matched.csv")
    ap.add_argument("--beta", type=float, default=HMMConfig.beta)
    ap.add_argument("--turn-weight", type=float, default=HMMConfig.turn_weight)
    a = ap.parse_args()

    g = load_graph(a.graph)
    cfg = HMMConfig(beta=a.beta, turn_weight=a.turn_weight)
    trips = load_candidates(a.candidates)
    routes_path = a.out.rsplit(".", 1)[0] + "_routes.csv"
    with open(a.out, "w", newline="") as fo, open(routes_path, "w", newline="") as fr:
        wo, wr = csv.writer(fo), csv.writer(fr)
        wo.writerow(["trip_id", "point_idx", "road_id"])
        wr.writerow(["trip_id", "route"])
        for tid, pts in trips.items():
            res = match_trip(pts, g, cfg)
            for p, rid in zip(pts, res.road_ids):
                wo.writerow([tid, p["point_idx"], rid])
            wr.writerow([tid, " ".join(res.route)])
    print(f"matched {len(trips)} trips -> {a.out}, {routes_path}")


if __name__ == "__main__":
    main()
