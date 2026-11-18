"""Small geometry helpers (stdlib only)."""
import math

EARTH_R = 6371000.0


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_R * math.asin(math.sqrt(a))


def bearing(lat1, lon1, lat2, lon2):
    """Initial bearing in degrees [0, 360)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def angle_diff(a, b):
    """Absolute smallest difference between two bearings, in [0, 180]."""
    d = abs(a - b) % 360.0
    return 360.0 - d if d > 180.0 else d


def _to_xy(lat, lon, lat0):
    """Local equirectangular projection (metres) - fine at city scale."""
    k = math.pi / 180.0 * EARTH_R
    return lon * k * math.cos(math.radians(lat0)), lat * k


def project_on_polyline(lat, lon, coords):
    """Project a point onto a polyline [(lat, lon), ...].

    Returns (distance_along_m, total_length_m, (proj_lat, proj_lon)).
    """
    lat0 = coords[0][0]
    px, py = _to_xy(lat, lon, lat0)
    best = (float("inf"), 0.0, coords[0])
    run = 0.0
    for (la1, lo1), (la2, lo2) in zip(coords[:-1], coords[1:]):
        x1, y1 = _to_xy(la1, lo1, lat0)
        x2, y2 = _to_xy(la2, lo2, lat0)
        dx, dy = x2 - x1, y2 - y1
        seg = math.hypot(dx, dy)
        t = 0.0 if seg == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg ** 2))
        qx, qy = x1 + t * dx, y1 + t * dy
        d = math.hypot(px - qx, py - qy)
        if d < best[0]:
            best = (d, run + t * seg, (la1 + t * (la2 - la1), lo1 + t * (lo2 - lo1)))
        run += seg
    return best[1], run, best[2]
