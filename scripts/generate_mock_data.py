#!/usr/bin/env python3
"""CLI wrapper around pathsnap.data.build_features.make_mock_features.

    python scripts/generate_mock_data.py --out data/processed/candidates_mock.csv
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pathsnap.data.build_features import make_mock_features  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())["mock"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/candidates_mock.csv")
    ap.add_argument("--n-points", type=int, default=cfg["n_points"])
    ap.add_argument("--n-roads", type=int, default=cfg["n_roads"])
    ap.add_argument("--seed", type=int, default=cfg["seed"])
    a = ap.parse_args()

    df = make_mock_features(a.n_points, a.n_roads, a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} rows ({df[['trip_id', 'point_idx']].drop_duplicates().shape[0]} points) to {out}")


if __name__ == "__main__":
    main()
