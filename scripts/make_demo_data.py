"""Generate a few synthetic trips in the raw PathSnap format, to test the API
before the real dataset is wired in:  python scripts/make_demo_data.py data/demo
"""
import sys
from pathlib import Path

import numpy as np


def make_trip(out: Path, tid: str, seed: int):
    rng = np.random.default_rng(seed)
    n = 8  # 8x8 grid, ~110 m spacing, around Bengaluru
    lon0, lat0, step = 77.53, 12.93, 0.001
    nodes = np.array([[lon0 + c * step, lat0 + r * step] for r in range(n) for c in range(n)])
    arcs = []
    for r in range(n):
        for c in range(n):
            u = r * n + c
            if c + 1 < n:
                arcs += [(u, u + 1), (u + 1, u)]
            if r + 1 < n:
                arcs += [(u, u + n), (u + n, u)]
    arcs = np.array(arcs)
    arc_of = {(a, b): i for i, (a, b) in enumerate(arcs)}
    # random monotone walk from the bottom-left corner
    r = c = 0
    path_nodes = [0]
    while r < n - 1 or c < n - 1:
        if c < n - 1 and (r == n - 1 or rng.random() < 0.5):
            c += 1
        else:
            r += 1
        path_nodes.append(r * n + c)
    route = [arc_of[(a, b)] for a, b in zip(path_nodes[:-1], path_nodes[1:])]
    track, t = [], 0.0
    for a, b in zip(path_nodes[:-1], path_nodes[1:]):
        for f in np.linspace(0, 1, 10, endpoint=False):
            p = nodes[a] + f * (nodes[b] - nodes[a]) + rng.normal(0, 0.00008, 2)  # ~9 m noise
            track.append([p[0], p[1], t])
            t += 1
    out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / f"{tid}.nodes", nodes, fmt="%.7f")
    np.savetxt(out / f"{tid}.arcs", arcs, fmt="%d")
    np.savetxt(out / f"{tid}.track", np.array(track), fmt=["%.7f", "%.7f", "%.0f"])
    np.savetxt(out / f"{tid}.route", np.array(route), fmt="%d")


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/demo")
    for k in range(3):
        make_trip(root / "evaluator_test", f"demo_{k:02d}", seed=k)
    print(f"Wrote 3 demo trips to {root}/evaluator_test")
