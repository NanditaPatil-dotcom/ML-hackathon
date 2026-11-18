
"""
PathSnap Evaluator Interface (Laptop 2):
Streamlit UI communicating with Laptop 1 (FastAPI backend).
Sends trip evaluation requests, displays metrics, and renders
interactive Folium maps.
"""

import streamlit as st
import requests
import folium
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="PathSnap Evaluator Dashboard",
    page_icon="🗺️",
    layout="wide"
)

# --------------------------------------------------
# PROJECT PATH
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 12px 16px;
        border: 1px solid #334155;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

st.title("🗺️ PathSnap: Distributed Map-Matching & Route Reconstruction")
st.caption(
    "PES University | UE24CS352A Machine Learning | Evaluation Console"
)

# --------------------------------------------------
# SIDEBAR CONFIGURATION
# --------------------------------------------------

st.sidebar.header("🔌 Network & API Configuration")

api_url = st.sidebar.text_input(
    "Laptop 1 API Server URL",
    value="http://10.0.15.134:8000"
).strip().rstrip("/")

fallback_trips = [
    "00000015",
    "00000017",
    "00000023",
    "00000030",
    "00000049",
    "00000059",
    "00000067",
    "00000073",
    "00000087",
    "00000094"
]

trips_list = fallback_trips

try:
    resp = requests.get(
        f"{api_url}/trips",
        timeout=5
    )

    if resp.status_code == 200:
        fetched = resp.json().get("trips", [])

        if fetched:
            trips_list = fetched

        st.sidebar.success("Connected to Laptop 1 API Server")

    else:
        st.sidebar.warning(
            f"API returned status code {resp.status_code}"
        )

except requests.RequestException as e:
    st.sidebar.error(
        f"Cannot reach Laptop 1 API Server: {e}"
    )

selected_trip = st.sidebar.selectbox(
    "Select Evaluation Trip ID:",
    trips_list
)

run_button = st.sidebar.button(
    "🚀 Run Distributed Inference & Evaluate",
    type="primary"
)

st.markdown("---")

# --------------------------------------------------
# EVALUATION REQUEST
# --------------------------------------------------

if run_button:

    with st.spinner(
        f"Processing Trip {selected_trip}..."
    ):

        try:
            res = requests.post(
                f"{api_url}/evaluate/{selected_trip}",
                timeout=120
            )

            if res.status_code == 200:
                st.session_state["last_result"] = res.json()
                st.session_state["last_trip"] = selected_trip

            else:
                st.error(
                    f"API Error ({res.status_code}): {res.text}"
                )

                st.session_state.pop("last_result", None)
                st.session_state.pop("last_trip", None)

        except requests.RequestException as e:
            st.error(f"Failed to communicate with API: {e}")

            st.session_state.pop("last_result", None)
            st.session_state.pop("last_trip", None)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

if "last_result" in st.session_state:

    data = st.session_state["last_result"]

    metrics = data.get("metrics", {})

    # Convert metric values safely to floats
    hmm_acc = float(metrics.get("hmm_top1_acc", 0) or 0)
    clf_acc = float(metrics.get("clf_top1_acc", 0) or 0)

    hmm_cont = float(metrics.get("hmm_continuity", 0) or 0)
    clf_cont = float(metrics.get("clf_continuity", 0) or 0)

    acc_delta = hmm_acc - clf_acc
    cont_delta = hmm_cont - clf_cont

    # --------------------------------------------------
    # KPI METRICS
    # --------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="HMM Top-1 Accuracy",
            value=f"{hmm_acc:.2f}%",
            delta=f"{acc_delta:+.2f}% vs Classifier"
        )

    with col2:
        st.metric(
            label="Route Continuity",
            value=f"{hmm_cont:.2f}%",
            delta=f"{cont_delta:+.2f}% vs Classifier"
        )

    with col3:
        st.metric(
            label="GPS Points Processed",
            value=f"{data.get('num_gps_observations', 0):,}"
        )

    with col4:
        processing_time = float(
            data.get("processing_time_sec", 0) or 0
        )

        st.metric(
            label="Inference Latency",
            value=f"{processing_time:.3f} s"
        )

    # --------------------------------------------------
    # MAP VISUALIZATION
    # --------------------------------------------------

    st.markdown("### 🗺️ Trajectory Reconstruction Visualization")
    st.caption("Ground-truth reference remains secure on Laptop 1 server. Below is the HMM reconstructed route overlaid on raw GPS.")

    gps_coords = data.get("gps_coords", [])
    pred_segments = data.get("predicted_segments", [])

    if gps_coords:
        center_lat = sum(p[0] for p in gps_coords) / len(gps_coords)
        center_lon = sum(p[1] for p in gps_coords) / len(gps_coords)
    else:
        center_lat, center_lon = 49.85, 8.59

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="OpenStreetMap"
    )

    # --------------------------------------------------
    # PREDICTED ROUTE (HMM Viterbi Reconstructed Path)
    # --------------------------------------------------

    fg_pred = folium.FeatureGroup(
        name="Predicted Route - HMM Viterbi (Purple)",
        show=True
    )

    for seg in pred_segments:
        if seg:
            folium.PolyLine(
                seg,
                color="#8b5cf6",
                weight=4,
                opacity=0.9
            ).add_to(fg_pred)

    fg_pred.add_to(m)

    # --------------------------------------------------
    # RAW GPS TRAJECTORY
    # --------------------------------------------------

    fg_gps = folium.FeatureGroup(
        name="Raw GPS Trajectory (Red)",
        show=True
    )

    if gps_coords:
        folium.PolyLine(
            gps_coords,
            color="#ef4444",
            weight=2,
            dash_array="4, 4",
            opacity=0.6
        ).add_to(fg_gps)

        sampling_step = max(1, len(gps_coords) // 80)
        for point in gps_coords[::sampling_step]:
            folium.CircleMarker(
                location=point,
                radius=3,
                color="#ef4444",
                fill=True,
                fill_color="#ef4444"
            ).add_to(fg_gps)

    fg_gps.add_to(m)

    # --------------------------------------------------
    # MAP CONTROLS
    # --------------------------------------------------

    folium.LayerControl(
        collapsed=False
    ).add_to(m)

    # --------------------------------------------------
    # SAVE MAP HTML
    # --------------------------------------------------

    out_html = (
        PROJECT_ROOT
        / "outputs"
        / "maps"
        / "latest_evaluation_map.html"
    )

    out_html.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    m.save(str(out_html))

    st.success(
        f"Map successfully saved to: {out_html}"
    )

    # --------------------------------------------------
    # DISPLAY INTERACTIVE MAP
    # --------------------------------------------------

    map_html = out_html.read_text(
        encoding="utf-8"
    )

    st.components.v1.html(
        map_html,
        height=600,
        scrolling=True
    )

else:

    st.info(
        "👈 Select a Trip ID from the sidebar and click "
        "'Run Distributed Inference & Evaluate' to test."
    )
