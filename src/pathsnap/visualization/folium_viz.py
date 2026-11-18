"""
PathSnap Folium Visualization:
Generates interactive HTML maps for trips showing:
1. Raw noisy GPS observations (markers + trajectory)
2. Road network segments along the reference route
3. Layer control, popup inspection, and legend
"""

import os
import folium
import numpy as np
from pathsnap.data.legacy_eda import load_trip_track, load_nodes, load_arcs, load_route

DATA_DIR = "/home/wingsfry/Projects/ML hackathon/Student folder"
MAPS_DIR = "/home/wingsfry/Projects/ML hackathon/outputs/maps"

def generate_trip_map(trip_id="00000001", partition="train", max_points=1000):
    os.makedirs(MAPS_DIR, exist_ok=True)
    trip_dir = os.path.join(DATA_DIR, partition, trip_id)
    
    track = load_trip_track(trip_dir, trip_id)
    nodes = load_nodes(trip_dir, trip_id)
    arcs = load_arcs(trip_dir, trip_id)
    route = load_route(trip_dir, trip_id)
    
    if track is None or nodes is None or arcs is None:
        print(f"Error: Missing files for trip {trip_id}")
        return None
        
    # Center map on median GPS observation
    # Note: track[:, 0] is lon, track[:, 1] is lat
    center_lat = float(np.median(track[:, 1]))
    center_lon = float(np.median(track[:, 0]))
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="OpenStreetMap")
    
    # 1. Layer for Ground Truth Route
    route_fg = folium.FeatureGroup(name="Ground Truth Route (Reference Arcs)", show=True)
    if route is not None:
        route_coords = []
        for arc_id in route:
            if arc_id < len(arcs):
                u, v = arcs[arc_id]
                if u < len(nodes) and v < len(nodes):
                    p_u = [float(nodes[u, 1]), float(nodes[u, 0])] # lat, lon
                    p_v = [float(nodes[v, 1]), float(nodes[v, 0])]
                    route_coords.append((p_u, p_v))
                    
        for p_u, p_v in route_coords:
            folium.PolyLine(
                [p_u, p_v],
                color="#28a745", # Green
                weight=5,
                opacity=0.85,
                tooltip=f"Road Segment"
            ).add_to(route_fg)
            
    route_fg.add_to(m)
    
    # 2. Layer for Raw GPS Observations
    gps_fg = folium.FeatureGroup(name="Raw GPS Observations (Noisy)", show=True)
    step = max(1, len(track) // max_points)
    sampled_track = track[::step]
    
    # GPS polyline
    gps_coords = [[float(p[1]), float(p[0])] for p in sampled_track]
    folium.PolyLine(
        gps_coords,
        color="#dc3545", # Red dashed
        weight=2,
        dash_array="5, 5",
        opacity=0.65
    ).add_to(gps_fg)
    
    # GPS CircleMarkers (subsample for rendering performance)
    for i, p in enumerate(sampled_track[::max(1, len(sampled_track)//300)]):
        folium.CircleMarker(
            location=[float(p[1]), float(p[0])],
            radius=3,
            color="#007bff", # Blue
            fill=True,
            fill_color="#007bff",
            fill_opacity=0.7,
            popup=f"GPS Point #{i*step}<br>Time: {p[2]:.0f}s<br>Lat: {p[1]:.5f}<br>Lon: {p[0]:.5f}"
        ).add_to(gps_fg)
        
    gps_fg.add_to(m)
    
    # Add Layer Control
    folium.LayerControl(collapsed=False).add_to(m)
    
    out_file = os.path.join(MAPS_DIR, f"trip_{trip_id}_visualization.html")
    m.save(out_file)
    print(f"Saved interactive Folium map to: {out_file}")
    return out_file

if __name__ == "__main__":
    # Generate map for trip 00000001
    generate_trip_map("00000001", "train")
    # Generate map for student test trip 00000000
    generate_trip_map("00000000", "student_test")
