"""
PathSnap Evaluator Interface (Laptop 2):
Streamlit UI that communicates with Laptop 1 (FastAPI backend),
sends trip evaluation requests, displays metrics, and renders interactive Folium maps.
"""

import streamlit as st
import requests
import folium

st.set_page_config(
    page_title="PathSnap Evaluator Dashboard",
    page_icon="🗺️",
    layout="wide"
)

# Header
st.title("🗺️ PathSnap: Distributed Map-Matching & Route Reconstruction")
st.caption("PES University | UE24CS352A Machine Learning | Evaluation Console")

# Sidebar Configuration
st.sidebar.header("🔌 Network & API Configuration")
api_url = st.sidebar.text_input("Laptop 1 API Server URL", value="http://localhost:8000")

# Fetch trip list from API or use standard test trips
fallback_trips = [
    "00000000", "00000007", "00000009", "00000040", "00000052",
    "00000056", "00000065", "00000076", "00000077", "00000091"
]

trips_list = fallback_trips
try:
    resp = requests.get(f"{api_url}/trips", timeout=2)
    if resp.status_code == 200:
        fetched = resp.json().get("trips", [])
        if fetched:
            trips_list = fetched
        st.sidebar.success("Connected to Laptop 1 API Server")
    else:
        st.sidebar.warning("API reachable but returned non-200 status")
except Exception:
    st.sidebar.error("Cannot reach Laptop 1. Verify IP address & Wi-Fi connection.")

selected_trip = st.sidebar.selectbox("Select Evaluation Trip ID:", trips_list)

run_button = st.sidebar.button("🚀 Run Distributed Inference & Evaluate", type="primary")

st.markdown("---")

if run_button or "last_result" in st.session_state:
    if run_button:
        with st.spinner(f"Sending evaluation request for Trip {selected_trip} to Laptop 1..."):
            try:
                res = requests.post(f"{api_url}/evaluate/{selected_trip}", timeout=30)
                if res.status_code == 200:
                    data = res.json()
                    st.session_state["last_result"] = data
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
                    data = None
            except Exception as e:
                st.error(f"Failed to communicate with API server: {e}")
                data = None
    else:
        data = st.session_state.get("last_result", None)

    if data:
        metrics = data.get("metrics", {})
        
        # Display Metric KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        
        hmm_acc = metrics.get("hmm_top1_acc", 0.0)
        clf_acc = metrics.get("clf_top1_acc", 0.0)
        acc_delta = hmm_acc - clf_acc
        
        hmm_cont = metrics.get("hmm_continuity", 0.0)
        clf_cont = metrics.get("clf_continuity", 0.0)
        cont_delta = hmm_cont - clf_cont
        
        with col1:
            st.metric(
                label="HMM Top-1 Accuracy",
                value=f"{hmm_acc:.2f}%" if isinstance(hmm_acc, float) else "N/A",
                delta=f"+{acc_delta:.2f}% vs Classifier" if acc_delta > 0 else f"{acc_delta:.2f}%"
            )
            
        with col2:
            st.metric(
                label="Route Continuity",
                value=f"{hmm_cont:.2f}%",
                delta=f"+{cont_delta:.2f}% vs Classifier" if cont_delta > 0 else f"{cont_delta:.2f}%"
            )
            
        with col3:
            st.metric(
                label="GPS Points Processed",
                value=f"{data.get('num_gps_observations', 0):,}"
            )
            
        with col4:
            st.metric(
                label="Inference Latency",
                value=f"{data.get('processing_time_sec', 0.0)} s"
            )
            
        st.markdown("### 🗺️ Trajectory Reconstruction Visualization")
        
        # Build Folium map
        gps_coords = data.get("gps_coords", [])
        gt_segments = data.get("ground_truth_segments", [])
        pred_segments = data.get("predicted_segments", [])
        
        if gps_coords:
            center_lat = sum(p[0] for p in gps_coords) / len(gps_coords)
            center_lon = sum(p[1] for p in gps_coords) / len(gps_coords)
        else:
            center_lat, center_lon = 49.85, 8.59
            
        m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="OpenStreetMap")
        
        # Ground truth route layer
        fg_gt = folium.FeatureGroup(name="Ground Truth Route (Green)", show=True)
        for seg in gt_segments:
            folium.PolyLine(seg, color="#10b981", weight=5, opacity=0.85).add_to(fg_gt)
        fg_gt.add_to(m)
        
        # Predicted reconstructed route layer
        fg_pred = folium.FeatureGroup(name="Predicted Route (Purple)", show=True)
        for seg in pred_segments:
            folium.PolyLine(seg, color="#8b5cf6", weight=4, opacity=0.9).add_to(fg_pred)
        fg_pred.add_to(m)
        
        # Raw GPS observations layer
        fg_gps = folium.FeatureGroup(name="Raw GPS Trajectory (Red)", show=True)
        if gps_coords:
            folium.PolyLine(gps_coords, color="#ef4444", weight=2, dash_array="4, 4", opacity=0.6).add_to(fg_gps)
            for p in gps_coords[::max(1, len(gps_coords)//80)]:
                folium.CircleMarker(location=p, radius=3, color="#ef4444", fill=True, fill_color="#ef4444").add_to(fg_gps)
        fg_gps.add_to(m)
        
        folium.LayerControl(collapsed=False).add_to(m)
        
        # Save HTML preview
        out_html = "/home/wingsfry/Projects/ML hackathon/outputs/maps/latest_evaluation_map.html"
        m.save(out_html)
        
        st.components.v1.html(open(out_html, 'r').read(), height=550)
else:
    st.info("👈 Select a Trip ID from the sidebar and click 'Run Distributed Inference & Evaluate' to test.")
