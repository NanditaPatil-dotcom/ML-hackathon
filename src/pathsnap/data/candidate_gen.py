"""
PathSnap Candidate Generation:
Uses spatial indexing (cKDTree) to retrieve candidate road segments
within a search radius for each GPS observation in a trip.
"""

import numpy as np
from scipy.spatial import cKDTree

def project_coordinates(coords, ref_lat):
    """
    Projects (lon, lat) angular coordinates into local metric Cartesian coordinates (x, y in meters).
    """
    cos_lat = np.cos(np.radians(ref_lat))
    m_per_deg_lat = 111132.954
    m_per_deg_lon = 111412.84 * cos_lat
    
    x = coords[:, 0] * m_per_deg_lon
    y = coords[:, 1] * m_per_deg_lat
    return np.column_stack([x, y]), m_per_deg_lon, m_per_deg_lat

def build_road_spatial_index(nodes, arcs, ref_lat):
    """
    Builds a cKDTree on road segment midpoints.
    """
    node_xy, m_lon, m_lat = project_coordinates(nodes, ref_lat)
    
    u_idx = arcs[:, 0]
    v_idx = arcs[:, 1]
    
    p_u = node_xy[u_idx]
    p_v = node_xy[v_idx]
    
    midpoints = (p_u + p_v) / 2.0
    lengths = np.linalg.norm(p_v - p_u, axis=1)
    
    tree = cKDTree(midpoints)
    return tree, node_xy, p_u, p_v, lengths

def dist_point_to_segment(p, a, b):
    """
    Perpendicular distance from point p to segment [a, b] in 2D meters.
    Returns (distance, projection_t).
    """
    v = b - a
    len_sq = np.sum(v**2)
    if len_sq < 1e-6:
        return np.linalg.norm(p - a), 0.0
    t = np.clip(np.dot(p - a, v) / len_sq, 0.0, 1.0)
    proj = a + t * v
    return float(np.linalg.norm(p - proj)), float(t)

def generate_candidates_for_trip(track, nodes, arcs, search_radius=75.0, top_k=8):
    """
    Generates candidate road segments for each GPS point in the track.
    Returns:
        candidates_per_obs: dict mapping obs_idx -> list of candidate dicts
    """
    if track is None or len(track) == 0:
        return {}
        
    ref_lat = float(np.mean(track[:, 1]))
    tree, node_xy, p_u, p_v, lengths = build_road_spatial_index(nodes, arcs, ref_lat)
    track_xy, _, _ = project_coordinates(track[:, :2], ref_lat)
    
    # Query spatial tree
    dists, indices = tree.query(track_xy, k=top_k, distance_upper_bound=search_radius)
    
    num_arcs = len(arcs)
    candidates_per_obs = {}
    
    # Precompute GPS heading
    gps_headings = np.zeros(len(track))
    gps_speeds = np.zeros(len(track))
    
    for i in range(len(track)):
        if i == 0 and len(track) > 1:
            dp = track_xy[1] - track_xy[0]
            dt = max(track[1, 2] - track[0, 2], 0.1)
        elif i > 0:
            dp = track_xy[i] - track_xy[i - 1]
            dt = max(track[i, 2] - track[i - 1, 2], 0.1)
        else:
            dp = np.array([0.0, 0.0])
            dt = 1.0
            
        dist_step = np.linalg.norm(dp)
        gps_speeds[i] = dist_step / dt
        if dist_step > 0.5:
            gps_headings[i] = np.arctan2(dp[1], dp[0])
        elif i > 0:
            gps_headings[i] = gps_headings[i - 1] # retain heading if stopped
            
    for i, p in enumerate(track_xy):
        cand_list = []
        raw_cand_indices = indices[i]
        
        # If no candidates within search radius, take the 3 closest regardless of distance
        valid_indices = [idx for idx in raw_cand_indices if idx < num_arcs]
        if len(valid_indices) == 0:
            _, fallback_indices = tree.query(p, k=min(3, num_arcs))
            valid_indices = [idx for idx in fallback_indices if idx < num_arcs]
            
        for rank, arc_id in enumerate(valid_indices):
            a_start = p_u[arc_id]
            a_end = p_v[arc_id]
            
            perp_dist, t_ratio = dist_point_to_segment(p, a_start, a_end)
            arc_vec = a_end - a_start
            arc_heading = np.arctan2(arc_vec[1], arc_vec[0])
            
            # Heading difference in radians wrapped to [-pi, pi]
            heading_diff = np.abs(np.arctan2(np.sin(gps_headings[i] - arc_heading),
                                             np.cos(gps_headings[i] - arc_heading)))
            cos_heading = np.cos(heading_diff)
            
            cand_list.append({
                "obs_idx": i,
                "arc_id": int(arc_id),
                "dist_to_segment": perp_dist,
                "projection_ratio": t_ratio,
                "heading_diff": float(heading_diff),
                "cos_heading": float(cos_heading),
                "segment_length": float(lengths[arc_id]),
                "gps_speed": float(gps_speeds[i]),
                "rank_by_midpoint": rank + 1,
            })
            
        # Re-sort candidates by exact perpendicular distance
        cand_list.sort(key=lambda c: c["dist_to_segment"])
        for r, c in enumerate(cand_list):
            c["rank_by_dist"] = r + 1
            
        candidates_per_obs[i] = cand_list
        
    return candidates_per_obs
