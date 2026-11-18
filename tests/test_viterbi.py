from hmm import RoadGraph, match_trip
from hmm.mock import TRUE_ROUTE, build_mock_graph, simulate_trip


def line_graph(oneway_b=False):
    # a --A--> b --B--> c, plus a parallel decoy road D from x to y
    nodes = {"a": (0.0, 0.0), "b": (0.0, 0.001), "c": (0.0, 0.002),
             "x": (0.0005, 0.0), "y": (0.0005, 0.002)}
    segs = [{"road_id": "A", "u": "a", "v": "b"},
            {"road_id": "B", "u": "b", "v": "c", "oneway": oneway_b},
            {"road_id": "D", "u": "x", "v": "y"}]
    return RoadGraph(nodes, segs)


def test_viterbi_overrides_noisy_emission():
    g = line_graph()
    # point 1's model prefers the disconnected decoy D, but A->B is the only consistent route
    pts = [{"lat": 0.0, "lon": 0.0003, "candidates": [("A", 0.9), ("D", 0.1)]},
           {"lat": 0.0002, "lon": 0.0010, "candidates": [("D", 0.6), ("B", 0.4)]},
           {"lat": 0.0, "lon": 0.0017, "candidates": [("B", 0.9), ("D", 0.1)]}]
    res = match_trip(pts, g)
    assert res.road_ids == ["A", "B", "B"]
    assert res.route == ["A", "B"]
    assert res.breaks == []


def test_oneway_blocks_reverse_travel():
    g = line_graph(oneway_b=True)
    d, _, _ = g.route("B", 50.0, "A", 50.0)
    assert d == float("inf")
    d2, _, _ = line_graph(oneway_b=False).route("B", 50.0, "A", 50.0)
    assert d2 < float("inf")


def test_gap_is_filled_with_intermediate_roads():
    g = RoadGraph.from_dict(build_mock_graph())
    # first point on H0_0, next point on H0_2 -> route must pass H0_1
    pts = [[("H0_0", 1.0)], [("H0_2", 1.0)]]   # connectivity-only mode (no GPS coords)
    res = match_trip(pts, g)
    assert res.route == ["H0_0", "H0_1", "H0_2"]


def test_break_when_unreachable():
    g = line_graph()
    pts = [{"lat": 0.0, "lon": 0.0003, "candidates": [("A", 1.0)]},
           {"lat": 0.0005, "lon": 0.0005, "candidates": [("D", 1.0)]}]
    res = match_trip(pts, g)
    assert res.road_ids == ["A", "D"]
    assert res.breaks == [1]


def test_points_without_candidates_are_skipped():
    g = line_graph()
    pts = [{"lat": 0.0, "lon": 0.0003, "candidates": [("A", 1.0)]},
           {"lat": 0.0, "lon": 0.0008, "candidates": []},
           {"lat": 0.0, "lon": 0.0015, "candidates": [("B", 1.0)]}]
    assert match_trip(pts, g).road_ids == ["A", None, "B"]


def test_mock_trip_beats_argmax():
    g = RoadGraph.from_dict(build_mock_graph())
    a = b = 0
    for seed in range(20):
        pts, truth = simulate_trip(g, noise_m=25, seed=seed)
        res = match_trip(pts, g)
        greedy = [max(p["candidates"], key=lambda c: c[1])[0] for p in pts]
        a += sum(x == y for x, y in zip(greedy, truth))
        b += sum(x == y for x, y in zip(res.road_ids, truth))
    assert b > a
    pts, _ = simulate_trip(g, noise_m=25, seed=3)
    assert match_trip(pts, g).route == TRUE_ROUTE
