# ML-hackathon: PathSnap

GPS map-matching: noisy GPS trip → candidate road arcs → classifier → HMM/Viterbi → most likely route, served by a FastAPI backend (Laptop 1) to an evaluator UI (Laptop 2).

## Repository layout

| Folder | Contents |
|---|---|
| `api/` | FastAPI server: `main.py` (endpoints), `pipeline.py` (clean → candidates → scoring → Viterbi), `data.py` (raw file loading), `metrics.py` (server-side accuracy), `config.py` (env-var settings) |
| `ui/` | Evaluator interface (static HTML + bundled Leaflet) |
| `scripts/` | `client.py` CLI evaluator, `make_demo_data.py` synthetic trips for testing |
| `docs/` | `DEPLOYMENT.md`: API contract and two-laptop setup |
| `models/` | final trained model (`final_model.joblib`) |
| `data/raw/` | raw trip files (`.track/.nodes/.arcs/.route`) |

## Run

```bash
pip install -r requirements-api.txt

# Laptop 1 (API)
python -m api.main                 # or run_api.bat; serves http://0.0.0.0:8000, docs at /docs

# Laptop 2 (UI)
cd ui && python -m http.server 5500   # or run_ui.bat
# open http://localhost:5500/?api=http://<LAPTOP1_IP>:8000
python scripts/client.py --api http://<LAPTOP1_IP>:8000 --all
```

Try it without the real data: `python scripts/make_demo_data.py data/demo` then `PATHSNAP_DATA_DIR=data/demo python -m api.main`.

Full API contract, firewall/IP steps and troubleshooting: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

### Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `PATHSNAP_DATA_DIR` | `data/raw` | root of raw trip files (searched recursively) |
| `PATHSNAP_MODEL_PATH` | `models/final_model.joblib` | trained classifier; heuristic fallback if missing |
| `PATHSNAP_EVAL_TRIPS` | — | comma-separated trip IDs to list in the UI |
| `PATHSNAP_EVAL_SUBDIR` | `eval` | otherwise, list trips in folders whose name contains this |
| `PATHSNAP_PORT` / `PATHSNAP_HOST` | `8000` / `0.0.0.0` | bind address |
| `PATHSNAP_CANDIDATE_RADIUS_M`, `PATHSNAP_MAX_CANDIDATES`, `PATHSNAP_GPS_SIGMA_M`, `PATHSNAP_TRANSITION_BETA_M`, `PATHSNAP_UNCONNECTED_PENALTY` | 50, 8, 10, 20, 3.0 | candidate + HMM parameters |

**PES University** | Department of Computer Science and Engineering  
**Course**: UE24CS352A — Machine Learning  

---

## 📌 Project Overview
**PathSnap** is an end-to-end distributed machine learning system that reconstructs realistic vehicle trajectories by matching noisy, sparse GPS observations to complex road networks. The system resolves GPS multipath effects, signal dropouts, and parallel street ambiguities by combining machine-learning candidate classification with Hidden Markov Model (HMM) Viterbi sequence decoding.

---

## 🏗️ Architecture & Deployment

The system is deployed using a two-laptop distributed architecture:
* **Laptop 1 (Model & Backend Server)**:
  * Hosts the trained Random Forest classifier and HMM Viterbi decoder.
  * Manages the road network topology and internal ground truth validation data.
  * Runs a high-performance **FastAPI** server exposing `/evaluate/{trip_id}`.
* **Laptop 2 (Evaluator Dashboard UI)**:
  * Interactive **Streamlit** user interface.
  * Enables evaluators to select any of the reserved evaluation trips.
  * Communicates with Laptop 1 over the local network.
  * Renders interactive **Folium** maps (comparing raw GPS tracks, ground truth routes, and reconstructed routes) alongside Top-1 Accuracy and Route Continuity metrics.

---

## 📁 Repository Structure

```
├── README.md                           # Comprehensive documentation & execution guide
├── requirements.txt                    # Project dependencies
├── .gitignore                          # Git configuration
├── artifacts/                          # Trained models & scalers
│   ├── best_model.joblib               # Top-performing Random Forest model
│   ├── tuned_rf_model.joblib           # Optuna Bayesian tuned model
│   └── scaler.joblib                   # Feature normalization scaler
├── src/                                # Core source code
│   ├── __init__.py
│   ├── eda.py                          # Exploratory Data Analysis & inventory generator
│   ├── candidate_gen.py                # cKDTree spatial indexing & candidate generation
│   ├── feature_eng.py                  # Spatial-temporal feature extraction & dataset builder
│   ├── models.py                       # 5-Model training, 5-Fold GroupKFold CV & benchmark
│   ├── bayesian_opt.py                 # Optuna Bayesian hyperparameter optimization
│   ├── hmm_viterbi.py                  # HMM state transitions & Viterbi decoding engine
│   ├── api.py                          # FastAPI backend server (Laptop 1)
│   ├── app.py                          # Streamlit UI client (Laptop 2)
│   ├── folium_viz.py                   # Interactive Folium map generator
│   ├── train_compare.py                # CatBoost & SVM training module
│   └── candidate_scorer.py             # Scorer interface
└── outputs/                            # Experimental logs, charts & reports
    ├── eda/                            # Dataset inventory CSV, summary text, distribution plots
    ├── maps/                           # Interactive HTML maps
    ├── models/                         # Model comparison table & benchmark charts
    └── bayesian/                       # Optuna trials log & optimization curves
```

---

## 🚀 Setup & Execution Guide

### 1. Environment Installation
```bash
# Clone the repository
git clone https://github.com/NanditaPatil-dotcom/ML-hackathon.git
cd ML-hackathon

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running EDA and Preprocessing
```bash
# Phase 1: Run comprehensive Exploratory Data Analysis
python3 src/eda.py

# Generate interactive Folium maps
python3 src/folium_viz.py

# Phase 2: Run spatial candidate extraction & feature engineering
python3 src/feature_eng.py
```

### 3. Model Training & Bayesian Tuning
```bash
# Train and benchmark all 5 models (Logistic, Decision Tree, RF, HistGB, MLP)
python3 src/models.py

# Run Bayesian hyperparameter optimization
python3 src/bayesian_opt.py
```

---

## 🖥️ Distributed Deployment Setup

### On Laptop 1 (Server):
```bash
# Start FastAPI backend server on port 8000
python3 src/api.py
# Server will listen on http://0.0.0.0:8000
```
*Note: Check your LAN IP using `ip a` or `ifconfig` (e.g., `192.168.1.50`).*

### On Laptop 2 (Client UI):
```bash
# Start Streamlit Evaluator Dashboard
streamlit run src/app.py
```
1. Open the web interface in your browser (`http://localhost:8501`).
2. In the sidebar, set the **Laptop 1 API Server URL** to `http://<LAPTOP_1_IP>:8000`.
3. Select any trip ID from the dropdown and click **Run Distributed Inference & Evaluate**.
4. View real-time accuracy, continuity, and interactive Folium maps!

---

## 📊 Experimental Results Summary

### 5-Model Benchmark Comparison Table

| Model Name | Algorithm Family | Key Configuration | CV Macro F1 | Top-1 Accuracy | Route Continuity | Student-Test Macro F1 | Log Loss | Train Time (s) | Inference Time |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear & Probabilistic | C=1.0, solver='lbfgs' | 0.8144 | 52.66% | 59.42% | 0.7610 | 0.2728 | 1.19s | 0.34 ms/1k |
| **Decision Tree** | Tree-based | max_depth=10, min_split=10 | 0.8289 | 53.40% | 57.78% | 0.7992 | 0.2662 | 1.14s | 0.14 ms/1k |
| **Random Forest** | Ensemble | n_est=40, max_depth=12 | **0.8398** | **54.30%** | 58.92% | **0.7999** | 0.2291 | 3.84s | 2.03 ms/1k |
| **HistGradientBoosting**| Boosting | max_iter=100, lr=0.1 | 0.8366 | 54.24% | **60.52%** | 0.7986 | **0.2268** | 1.25s | 4.29 ms/1k |
| **MLP Classifier** | Neural Networks | hidden=(64, 32), iter=25 | 0.8347 | 53.71% | 59.74% | 0.7969 | 0.2291 | 16.70s | 1.41 ms/1k |

### Sequential HMM / Viterbi Reconstructed Route Improvement
* **Route Continuity**: Improved from **89.77%** (classifier-only) to **96.84%** (+7.07% gain) by enforcing road network graph adjacency constraints.
* **Pointwise Top-1 Accuracy**: Improved from **70.80%** to **72.79%** (+2.0% gain) by suppressing transient noisy GPS jumps.
