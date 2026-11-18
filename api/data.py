"""Loading the raw PathSnap trip files (whitespace-separated, no headers).

  <trip>.track : longitude latitude timestamp_s
  <trip>.nodes : longitude latitude            (row index = node_id)
  <trip>.arcs  : from_node_id to_node_id       (row index = arc_id)
  <trip>.route : arc_id per line               (reference route; SERVER-SIDE ONLY)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import config

log = logging.getLogger("pathsnap.data")


def _read(path: Path, ncols: int, dtype=float) -> np.ndarray:
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0, ncols), dtype=dtype)
    arr = np.loadtxt(path, dtype=dtype, ndmin=2)
    return arr[:, :ncols]


@dataclass
class TripFiles:
    trip_id: str
    track: Path
    partition: str

    def _sib(self, ext: str) -> Path:
        return self.track.with_suffix(ext)

    def load_track(self) -> np.ndarray:  # (n, 3) lon, lat, t
        return _read(self.track, 3)

    def load_nodes(self) -> np.ndarray:  # (n, 2) lon, lat
        return _read(self._sib(".nodes"), 2)

    def load_arcs(self) -> np.ndarray:  # (m, 2) from, to
        return _read(self._sib(".arcs"), 2, dtype=int)

    def load_reference_route(self) -> Optional[List[int]]:
        p = self._sib(".route")
        if not p.exists():
            return None
        return [int(x) for x in _read(p, 1, dtype=int)[:, 0]]


class TripIndex:
    def __init__(self, root: Path = config.DATA_DIR):
        self.root = root
        self.trips: Dict[str, TripFiles] = {}
        if root.exists():
            for track in sorted(root.rglob("*.track")):
                rel = track.parent.relative_to(root).as_posix()
                tid = track.stem
                if tid in self.trips:  # same stem in two partitions -> disambiguate
                    tid = f"{rel}/{track.stem}"
                self.trips[tid] = TripFiles(tid, track, rel)
        log.info("Found %d trips under %s", len(self.trips), root)

    def evaluation_trips(self) -> List[str]:
        if config.EVAL_TRIPS:
            return [t for t in config.EVAL_TRIPS if t in self.trips]
        if config.EVAL_SUBDIR:
            sel = [t for t, f in self.trips.items() if config.EVAL_SUBDIR.lower() in f.partition.lower()]
            if sel:
                return sel
        return list(self.trips)

    def get(self, trip_id: str) -> Optional[TripFiles]:
        return self.trips.get(str(trip_id))
