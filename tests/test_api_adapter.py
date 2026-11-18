from types import SimpleNamespace

import numpy as np

from hmm.api_adapter import viterbi_for_api
from hmm.mock import build_mock_graph, simulate_trip
from hmm import RoadGraph


def test_adapter_matches_api_shapes():
    d = build_mock_graph()
    names = list(d["nodes"])
    nodes = np.array([[d["nodes"][n][1], d["nodes"][n][0]] for n in names])  # lon, lat
    arcs, arc_of = [], {}
    for s in d["segments"]:  # two-way -> two directed arcs
        u, v = names.index(s["u"]), names.index(s["v"])
        arc_of[s["road_id"]] = len(arcs); arcs.append([u, v])
        if not s.get("oneway"):
            arcs.append([v, u])
    net = SimpleNamespace(nodes=nodes, arcs=np.array(arcs))
    pts, truth = simulate_trip(RoadGraph.from_dict(d), noise_m=10, seed=3)
    track = np.array([[p["lon"], p["lat"], 5.0 * i] for i, p in enumerate(pts)])
    layers = []
    for i, p in enumerate(pts):
        ids = np.array([arc_of[r] for r, _ in p["candidates"]])
        layers.append(SimpleNamespace(obs_index=i, arc_ids=ids,
                                      scores=np.array([q for _, q in p["candidates"]])))
    path = viterbi_for_api(net, track, layers)
    assert len(path) == len(layers)
    chosen = [int(L.arc_ids[k]) for L, k in zip(layers, path)]
    assert chosen == [arc_of[r] for r in truth]
