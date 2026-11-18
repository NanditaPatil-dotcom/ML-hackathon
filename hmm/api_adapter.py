"""Drop-in for api/pipeline.py ``Pipeline.viterbi(net, track, layers)``.

    from hmm.api_adapter import viterbi_for_api
    path = viterbi_for_api(net, track, layers)   # -> chosen candidate index per layer

net    : api RoadNetwork (``.nodes`` (n,2) lon,lat ; ``.arcs`` (m,2) from,to ; arc_id = row index,
         arcs are directed, so a two-way street is two arcs)
track  : (N,3) lon, lat, t
layers : objects with ``.obs_index``, ``.arc_ids`` and ``.scores`` (classifier emission probs)
"""
from .graph import RoadGraph
from .matcher import HMMConfig, match_trip

_cache = {}


def graph_from_api_network(net):
    key = id(net)
    hit = _cache.get(key)
    if hit is not None and hit[0] is net:
        return hit[1]
    n_nodes = len(net.nodes)
    nodes = {i: (float(net.nodes[i][1]), float(net.nodes[i][0])) for i in range(n_nodes)}
    segs = []
    for aid, (u, v) in enumerate(net.arcs):
        u, v = int(u), int(v)
        if 0 <= u < n_nodes and 0 <= v < n_nodes:
            segs.append({"road_id": aid, "u": u, "v": v, "oneway": True})
    g = RoadGraph(nodes, segs)
    _cache.clear()
    _cache[key] = (net, g)
    return g


def viterbi_for_api(net, track, layers, config=None):
    if not layers:
        return []
    g = graph_from_api_network(net)
    points = []
    for L in layers:
        lon, lat = float(track[L.obs_index][0]), float(track[L.obs_index][1])
        points.append({"lat": lat, "lon": lon,
                       "candidates": [(int(a), float(s)) for a, s in zip(L.arc_ids, L.scores)]})
    res = match_trip(points, g, config or HMMConfig())
    out = []
    for L, rid in zip(layers, res.road_ids):
        ids = [int(a) for a in L.arc_ids]
        if rid is not None and rid in ids:
            out.append(ids.index(rid))
        else:  # no usable candidate for this layer: fall back to the best emission
            out.append(max(range(len(ids)), key=lambda k: float(L.scores[k])) if ids else 0)
    return out
