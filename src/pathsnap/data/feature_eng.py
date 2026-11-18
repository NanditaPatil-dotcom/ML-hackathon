"""
PathSnap Feature Engineering and Dataset Builder:
1. Projects track and route to match observations monotonically to true traversed arcs.
2. Combines candidate generation with discriminative features:
   - dist_to_segment (perpendicular distance)
   - projection_ratio (t along arc)
   - heading_diff and cos_heading
   - segment_length
   - gps_speed
   - rank_by_dist
   - dist_margin (difference between candidate dist and closest candidate dist)
   - relative_dist_ratio
3. Labels each candidate as 1 (True matching arc) or 0 (Alternative candidate).
4. Saves dataset ready for Model Training (5 models) and Evaluation.
"""

import os
import glob
import numpy as np
import pandas as pd
from pathsnap.data.legacy_eda import load_trip_track, load_nodes, load_arcs, load_route
from pathsnap.data.candidate_gen import (
    project_coordinates,
    dist_point_to_segment,
    generate_candidates_for_trip
)

DATA_DIR = "/home/wingsfry/Projects/ML hackathon/Student folder"
OUTPUT_DIR = "/home/wingsfry/Projects/ML hackathon/outputs"

def match_track_to_route(track, nodes, arcs, route):
    """
    Monotonically maps each GPS observation in track to the true active arc in route.
    Returns:
        true_arc_per_obs: dict mapping obs_idx -> true_arc_id
    """
    ref_lat = float(np.mean(track[:, 1]))
    node_xy, _, _ = project_coordinates(nodes, ref_lat)
    track_xy, _, _ = project_coordinates(track[:, :2], ref_lat)
    
    true_arc_per_obs = {}
    r_idx = 0
    route_len = len(route)
    
    for i, p in enumerate(track_xy):
        best_dist = float('inf')
        best_r = r_idx
        
        # Look around current route index within a forward-leaning window
        window = range(max(0, r_idx - 2), min(route_len, r_idx + 12))
        for r in window:
            arc_id = route[r]
            u, v = arcs[arc_id]
            dist, _ = dist_point_to_segment(p, node_xy[u], node_xy[v])
            if dist < best_dist:
                best_dist = dist
                best_r = r
                
        r_idx = best_r
        true_arc_per_obs[i] = int(route[best_r])
        
    return true_arc_per_obs

def extract_features_for_trip(trip_dir, trip_id, is_train=True):
    """
    Processes a single trip into a DataFrame of candidate feature vectors.
    """
    track = load_trip_track(trip_dir, trip_id)
    nodes = load_nodes(trip_dir, trip_id)
    arcs = load_arcs(trip_dir, trip_id)
    route = load_route(trip_dir, trip_id) if is_train else None
    
    if track is None or nodes is None or arcs is None:
        return None
        
    candidates_dict = generate_candidates_for_trip(track, nodes, arcs, search_radius=75.0, top_k=8)
    
    true_arcs = {}
    if is_train and route is not None:
        true_arcs = match_track_to_route(track, nodes, arcs, route)
        
    rows = []
    candidate_misses = 0
    total_obs = len(track)
    
    for obs_idx, cands in candidates_dict.items():
        true_arc = true_arcs.get(obs_idx, None)
        min_dist = cands[0]["dist_to_segment"] if cands else 0.0
        
        found_true = False
        for c in cands:
            is_true = 1 if (true_arc is not None and c["arc_id"] == true_arc) else 0
            if is_true:
                found_true = True
                
            dist_margin = c["dist_to_segment"] - min_dist
            relative_dist_ratio = c["dist_to_segment"] / (min_dist + 1e-4)
            
            rows.append({
                "trip_id": trip_id,
                "obs_idx": obs_idx,
                "arc_id": c["arc_id"],
                "dist_to_segment": c["dist_to_segment"],
                "projection_ratio": c["projection_ratio"],
                "heading_diff": c["heading_diff"],
                "cos_heading": c["cos_heading"],
                "segment_length": c["segment_length"],
                "gps_speed": c["gps_speed"],
                "rank_by_dist": c["rank_by_dist"],
                "dist_margin": dist_margin,
                "relative_dist_ratio": relative_dist_ratio,
                "label": is_true
            })
            
        if true_arc is not None and not found_true:
            candidate_misses += 1
            
    df_trip = pd.DataFrame(rows)
    return df_trip, candidate_misses, total_obs

def build_dataset(max_train_trips=20):
    """
    Builds the processed training dataset from training trips.
    """
    train_dir = os.path.join(DATA_DIR, "train")
    trip_ids = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    
    selected_trips = trip_ids[:max_train_trips]
    print(f"Building dataset from {len(selected_trips)} training trips...")
    
    all_dfs = []
    total_misses = 0
    total_obs_all = 0
    
    for i, tid in enumerate(selected_trips):
        trip_path = os.path.join(train_dir, tid)
        res = extract_features_for_trip(trip_path, tid, is_train=True)
        if res is not None:
            df_trip, misses, obs = res
            all_dfs.append(df_trip)
            total_misses += misses
            total_obs_all += obs
            if (i + 1) % 5 == 0 or (i + 1) == len(selected_trips):
                print(f"Processed {i + 1}/{len(selected_trips)} train trips. Total candidate samples: {sum(len(d) for d in all_dfs)}")
                
    final_df = pd.concat(all_dfs, ignore_index=True)
    out_csv = os.path.join(OUTPUT_DIR, "processed_train_features.csv")
    final_df.to_csv(out_csv, index=False)
    
    miss_rate = (total_misses / max(total_obs_all, 1)) * 100.0
    print(f"\nTrain dataset build complete!")
    print(f"Saved {len(final_df):,} candidate feature rows to {out_csv}")
    print(f"Candidate Miss Rate: {miss_rate:.2f}%")
    return final_df

def build_test_dataset():
    """
    Builds the processed test dataset from the 10 student_test trips.
    """
    test_dir = os.path.join(DATA_DIR, "student_test")
    trip_ids = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    
    print(f"Building test dataset from {len(trip_ids)} student_test trips...")
    all_dfs = []
    total_misses = 0
    total_obs_all = 0
    
    for i, tid in enumerate(trip_ids):
        trip_path = os.path.join(test_dir, tid)
        res = extract_features_for_trip(trip_path, tid, is_train=True)
        if res is not None:
            df_trip, misses, obs = res
            all_dfs.append(df_trip)
            total_misses += misses
            total_obs_all += obs
            
    final_df = pd.concat(all_dfs, ignore_index=True)
    out_csv = os.path.join(OUTPUT_DIR, "processed_test_features.csv")
    final_df.to_csv(out_csv, index=False)
    
    miss_rate = (total_misses / max(total_obs_all, 1)) * 100.0
    print(f"\nStudent Test dataset build complete!")
    print(f"Saved {len(final_df):,} candidate feature rows to {out_csv}")
    print(f"Student Test Candidate Miss Rate: {miss_rate:.2f}%")
    return final_df

if __name__ == "__main__":
    build_test_dataset()
