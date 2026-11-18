"""Geodesic helpers for PathSnap.

Longitude/latitude are angular coordinates, so plain Euclidean distance in
degrees is not an appropriate measure of physical distance (see Dataset
Exploration guidelines). We use two complementary approaches:

- Haversine great-circle distance for *point-to-point* distances where exact
  accuracy matters (speed/acceleration outlier filtering, transition
  distances). Cheap, vectorised, no external dependency.
- A local equirectangular (tangent-plane) projection, re-centered per trip on
  that trip's own road-network extent, for *point-to-segment* distance and
  for building a KD-tree spatial index over road segments. Trips here cover
  a few km², so the flat-earth approximation error is negligible (<0.1% at
  this scale) and far cheaper than a full geodesic projection per query.
"""
from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lon1, lat1, lon2, lat2):
    """Vectorised great-circle distance in metres. Inputs in degrees."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return EARTH_RADIUS_M * c


def bearing_deg(lon1, lat1, lon2, lat2):
    """Initial bearing (degrees, 0-360, 0 = north) from point 1 -> point 2."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    brng = np.degrees(np.arctan2(x, y))
    return (brng + 360.0) % 360.0


def bearing_diff_deg(b1, b2):
    """Smallest absolute difference between two bearings, 0-180 degrees."""
    d = np.abs(np.asarray(b1) - np.asarray(b2)) % 360.0
    return np.minimum(d, 360.0 - d)


class LocalProjector:
    """Equirectangular projection centred on a given (lon, lat) origin.

    x = R * cos(lat0) * (lon - lon0) [in radians], y = R * (lat - lat0)
    Both in metres. Good for local (few-km) extents, which is what a single
    trip's road network covers here.
    """

    def __init__(self, origin_lon: float, origin_lat: float):
        self.lon0 = np.radians(origin_lon)
        self.lat0 = np.radians(origin_lat)
        self.cos_lat0 = np.cos(self.lat0)

    def project(self, lon, lat):
        lon = np.radians(np.asarray(lon, dtype=float))
        lat = np.radians(np.asarray(lat, dtype=float))
        x = EARTH_RADIUS_M * self.cos_lat0 * (lon - self.lon0)
        y = EARTH_RADIUS_M * (lat - self.lat0)
        return x, y


def point_to_segment_distance_xy(px, py, ax, ay, bx, by):
    """Perpendicular distance (metres) from point(s) P to segment(s) A-B,
    all already in a local metric (x, y) projection. Vectorised over arrays.
    """
    px, py, ax, ay, bx, by = map(lambda v: np.asarray(v, dtype=float), (px, py, ax, ay, bx, by))
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    seg_len_sq = abx ** 2 + aby ** 2
    seg_len_sq = np.where(seg_len_sq == 0, 1e-9, seg_len_sq)
    t = (apx * abx + apy * aby) / seg_len_sq
    t = np.clip(t, 0.0, 1.0)
    proj_x = ax + t * abx
    proj_y = ay + t * aby
    return np.hypot(px - proj_x, py - proj_y)
