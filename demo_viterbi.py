"""Demo: HMM/Viterbi vs. per-point argmax on a mocked noisy trip.

    python demo_viterbi.py [--noise 35] [--seed 7]
"""
import argparse

from hmm import RoadGraph, match_trip
from hmm.mock import TRUE_ROUTE, build_mock_graph, simulate_trip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, default=25.0, help="GPS noise std (m)")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    g = RoadGraph.from_dict(build_mock_graph())
    pts, truth = simulate_trip(g, noise_m=args.noise, seed=args.seed)
    res = match_trip(pts, g)
    greedy = [max(p["candidates"], key=lambda c: c[1])[0] for p in pts]

    print(f"{'pt':>3} {'truth':>6} {'argmax':>7} {'viterbi':>8}  top candidates")
    for i, p in enumerate(pts):
        top = ", ".join(f"{r}:{q:.2f}" for r, q in p["candidates"][:3])
        flag = "" if res.road_ids[i] == truth[i] else "  <-- wrong"
        print(f"{i:>3} {truth[i]:>6} {greedy[i]:>7} {res.road_ids[i]:>8}  {top}{flag}")
    acc = lambda pred: sum(a == b for a, b in zip(pred, truth)) / len(truth)
    print(f"\nargmax accuracy : {acc(greedy):.0%}")
    print(f"viterbi accuracy: {acc(res.road_ids):.0%}")
    print(f"true route      : {TRUE_ROUTE}")
    print(f"viterbi route   : {res.route}")
    print(f"breaks          : {res.breaks}")

    # aggregate over many random trips
    a = b = n = 0
    for seed in range(50):
        p2, t2 = simulate_trip(g, noise_m=args.noise, seed=seed)
        r2 = match_trip(p2, g)
        g2 = [max(p["candidates"], key=lambda c: c[1])[0] for p in p2]
        a += sum(x == y for x, y in zip(g2, t2))
        b += sum(x == y for x, y in zip(r2.road_ids, t2))
        n += len(t2)
    print(f"\n50 random trips @ {args.noise:.0f} m noise: argmax {a/n:.1%}  ->  viterbi {b/n:.1%}")


if __name__ == "__main__":
    main()
