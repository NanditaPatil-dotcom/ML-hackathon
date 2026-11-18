"""
PathSnap EDA: Comprehensive Dataset Exploration
- Scans all trips in Train (80) and Student Test (10)
- Analyzes GPS track quality, sampling frequency, gaps
- Analyzes Road Network graph topology, degree distribution, and connectivity
- Generates summary statistics and CSV reports
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

DATA_DIR = "/home/wingsfry/Projects/ML hackathon/Student folder"
OUTPUT_DIR = "/home/wingsfry/Projects/ML hackathon/outputs/eda"

def load_trip_track(trip_dir, trip_id):
    track_path = os.path.join(trip_dir, f"{trip_id}.track")
    if not os.path.exists(track_path):
        return None
    data = []
    with open(track_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                data.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return np.array(data) if data else None

def load_nodes(trip_dir, trip_id):
    nodes_path = os.path.join(trip_dir, f"{trip_id}.nodes")
    if not os.path.exists(nodes_path):
        return None
    data = []
    with open(nodes_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                data.append((float(parts[0]), float(parts[1])))
    return np.array(data) if data else None

def load_arcs(trip_dir, trip_id):
    arcs_path = os.path.join(trip_dir, f"{trip_id}.arcs")
    if not os.path.exists(arcs_path):
        return None
    data = []
    with open(arcs_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                data.append((int(parts[0]), int(parts[1])))
    return np.array(data, dtype=int) if data else None

def load_route(trip_dir, trip_id):
    route_path = os.path.join(trip_dir, f"{trip_id}.route")
    if not os.path.exists(route_path):
        return None
    data = []
    with open(route_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                data.append(int(parts[0]))
    return np.array(data, dtype=int) if data else None

def run_eda():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    records = []
    
    partitions = [
        ("train", os.path.join(DATA_DIR, "train")),
        ("student_test", os.path.join(DATA_DIR, "student_test"))
    ]
    
    all_dts = []
    all_speeds = []
    
    for split_name, split_dir in partitions:
        trip_dirs = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
        print(f"Analyzing {split_name}: {len(trip_dirs)} trips...")
        
        for trip_id in trip_dirs:
            full_trip_dir = os.path.join(split_dir, trip_id)
            track = load_trip_track(full_trip_dir, trip_id)
            nodes = load_nodes(full_trip_dir, trip_id)
            arcs = load_arcs(full_trip_dir, trip_id)
            route = load_route(full_trip_dir, trip_id)
            
            num_points = len(track) if track is not None else 0
            num_nodes = len(nodes) if nodes is not None else 0
            num_arcs = len(arcs) if arcs is not None else 0
            num_route = len(route) if route is not None else 0
            
            # GPS analysis
            duration = 0.0
            dt_exact_1s_pct = 0.0
            gaps_count = 0
            avg_speed_mps = 0.0
            
            if track is not None and len(track) > 1:
                times = track[:, 2]
                dts = np.diff(times)
                all_dts.extend(dts.tolist())
                duration = times[-1] - times[0]
                dt_exact_1s_pct = float(np.mean(np.isclose(dts, 1.0, atol=1e-2))) * 100.0
                gaps_count = int(np.sum(dts > 2.0))
                
                # Approximate distance in meters
                # 1 deg lat ~ 111,000m, 1 deg lon ~ 111,000m * cos(lat)
                mean_lat = np.mean(track[:, 1])
                dx = np.diff(track[:, 0]) * 111000.0 * np.cos(np.radians(mean_lat))
                dy = np.diff(track[:, 1]) * 111000.0
                dist_m = np.sum(np.sqrt(dx**2 + dy**2))
                if duration > 0:
                    avg_speed_mps = dist_m / duration
                    all_speeds.append(avg_speed_mps)
            
            # Graph connectivity analysis
            in_degrees = {}
            out_degrees = {}
            if arcs is not None and len(arcs) > 0:
                for u, v in arcs:
                    out_degrees[u] = out_degrees.get(u, 0) + 1
                    in_degrees[v] = in_degrees.get(v, 0) + 1
                
                active_nodes = set(arcs[:, 0]).union(set(arcs[:, 1]))
                avg_out_degree = len(arcs) / max(len(active_nodes), 1)
            else:
                avg_out_degree = 0.0
                active_nodes = set()
            
            records.append({
                "partition": split_name,
                "trip_id": trip_id,
                "num_gps_points": num_points,
                "duration_sec": duration,
                "pct_1s_sampling": round(dt_exact_1s_pct, 2),
                "num_gaps_gt_2s": gaps_count,
                "avg_speed_mps": round(avg_speed_mps, 2),
                "num_nodes": num_nodes,
                "num_arcs": num_arcs,
                "active_network_nodes": len(active_nodes),
                "avg_out_degree": round(avg_out_degree, 2),
                "route_length_arcs": num_route,
                "lon_min": round(float(track[:, 0].min()), 6) if track is not None else None,
                "lon_max": round(float(track[:, 0].max()), 6) if track is not None else None,
                "lat_min": round(float(track[:, 1].min()), 6) if track is not None else None,
                "lat_max": round(float(track[:, 1].max()), 6) if track is not None else None,
            })
            
    df = pd.DataFrame(records)
    csv_path = os.path.join(OUTPUT_DIR, "trips_inventory.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved inventory of {len(df)} trips to {csv_path}")
    
    # Generate EDA Summary text
    summary_path = os.path.join(OUTPUT_DIR, "eda_summary.txt")
    with open(summary_path, "w") as f:
        f.write("================ PATHSNAP EXPLORATORY DATA ANALYSIS ================\n")
        f.write(f"Total Trips Analyzed: {len(df)}\n")
        f.write(f"Training Trips: {len(df[df['partition'] == 'train'])}\n")
        f.write(f"Student Test Trips: {len(df[df['partition'] == 'student_test'])}\n\n")
        
        f.write("--- GPS TRACK STATISTICS ---\n")
        f.write(f"Total GPS Observations: {df['num_gps_points'].sum():,}\n")
        f.write(f"Avg Observations/Trip: {df['num_gps_points'].mean():.1f} (std: {df['num_gps_points'].std():.1f})\n")
        f.write(f"Min/Max Observations: {df['num_gps_points'].min()} / {df['num_gps_points'].max()}\n")
        f.write(f"Avg Trip Duration: {df['duration_sec'].mean():.1f} seconds (~{df['duration_sec'].mean()/60:.1f} mins)\n")
        f.write(f"Sampling Regularity: {df['pct_1s_sampling'].mean():.2f}% of steps are exactly 1 second apart\n")
        f.write(f"Total Significant Gaps (>2s): {df['num_gaps_gt_2s'].sum()}\n")
        f.write(f"Avg Vehicle Speed: {df['avg_speed_mps'].mean()*3.6:.1f} km/h ({df['avg_speed_mps'].mean():.1f} m/s)\n\n")
        
        f.write("--- ROAD NETWORK STATISTICS ---\n")
        f.write(f"Avg Road Network Nodes per Trip: {df['num_nodes'].mean():.1f}\n")
        f.write(f"Avg Arcs (Directed Road Segments) per Trip: {df['num_arcs'].mean():.1f}\n")
        f.write(f"Avg Graph Out-Degree: {df['avg_out_degree'].mean():.2f}\n")
        f.write(f"Avg Route Length: {df['route_length_arcs'].mean():.1f} road segments\n")
    print(f"Saved summary report to {summary_path}")
    
    # Generate Visualizations
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(df["num_gps_points"], kde=True, color="royalblue", bins=20)
    plt.title("Distribution of GPS Points per Trip")
    plt.xlabel("Number of GPS Observations")
    plt.ylabel("Trip Count")
    
    plt.subplot(1, 2, 2)
    sns.histplot(df["num_arcs"], kde=True, color="seagreen", bins=20)
    plt.title("Distribution of Road Network Arcs per Trip")
    plt.xlabel("Number of Arcs")
    plt.ylabel("Trip Count")
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "dataset_distributions.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved plots to {plot_path}")

    return df

if __name__ == "__main__":
    run_eda()
