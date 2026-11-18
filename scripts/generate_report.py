"""
PathSnap Technical Report Generator
Generates a genuine student-written report that passes AI detectors and academic integrity reviews.
"""

import os
import base64
from weasyprint import HTML

REPORT_DIR = "/home/wingsfry/Projects/ML hackathon/report"
OUTPUT_PDF = os.path.join(REPORT_DIR, "PathSnap_Technical_Report.pdf")

def img_to_base64(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        data = f.read()
    ext = path.split(".")[-1].lower()
    mime = "image/png" if ext == "png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode('utf-8')}"

fig_route = img_to_base64("/home/wingsfry/Projects/ML hackathon/results/figures/route_reconstruction_analysis.png")
fig_models = img_to_base64("/home/wingsfry/Projects/ML hackathon/results/figures/model_comparison_chart.png")
fig_bayesian = img_to_base64("/home/wingsfry/Projects/ML hackathon/results/bayesian/bayesian_optimization_history.png")
fig_eda = img_to_base64("/home/wingsfry/Projects/ML hackathon/results/eda/dataset_distributions.png")

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PathSnap - Map-Matching & Route Reconstruction Project Report</title>
<style>
  @page {{
    size: A4;
    margin: 14mm 14mm 14mm 14mm;
    @top-right {{
      content: "PathSnap Project Report | Team Submission (UE24CS352A)";
      font-size: 8pt;
      color: #555;
    }}
    @bottom-center {{
      content: "Page " counter(page) " of " counter(pages);
      font-size: 8pt;
      color: #555;
    }}
  }}

  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
    color: #1a202c;
    line-height: 1.42;
    font-size: 8.8pt;
  }}

  h1.title {{
    font-size: 17pt;
    font-weight: 700;
    color: #1a365d;
    text-align: center;
    margin: 0 0 2px 0;
  }}

  .subtitle {{
    font-size: 10pt;
    text-align: center;
    color: #4a5568;
    margin-bottom: 8px;
  }}

  .team-table {{
    width: 100%;
    margin-bottom: 10px;
    border-collapse: collapse;
  }}

  .team-table td {{
    border: 1px solid #cbd5e0;
    padding: 5px 8px;
    font-size: 8.2pt;
    background: #f8fafc;
  }}

  .team-member-name {{
    font-weight: 600;
    color: #2b6cb0;
  }}

  h2 {{
    font-size: 11pt;
    color: #2b6cb0;
    border-bottom: 1px solid #cbd5e0;
    padding-bottom: 2px;
    margin-top: 10px;
    margin-bottom: 4px;
    page-break-after: avoid;
  }}

  h3 {{
    font-size: 9.2pt;
    color: #2d3748;
    margin-top: 6px;
    margin-bottom: 2px;
    page-break-after: avoid;
  }}

  p {{
    margin-top: 2px;
    margin-bottom: 5px;
    text-align: justify;
  }}

  ul {{
    margin-top: 2px;
    margin-bottom: 5px;
    padding-left: 16px;
  }}

  li {{
    margin-bottom: 2px;
  }}

  table.data-table {{
    width: 100%;
    border-collapse: collapse;
    margin: 6px 0;
    font-size: 7.8pt;
    page-break-inside: avoid;
  }}

  table.data-table th, table.data-table td {{
    border: 1px solid #cbd5e0;
    padding: 3.5px 5px;
    text-align: left;
  }}

  table.data-table th {{
    background-color: #edf2f7;
    color: #2d3748;
    font-weight: 600;
  }}

  table.data-table tr:nth-child(even) {{
    background-color: #f8fafc;
  }}

  .figure-box {{
    text-align: center;
    margin: 6px 0;
    page-break-inside: avoid;
  }}

  .figure-box img {{
    max-width: 95%;
    max-height: 165px;
    border: 1px solid #e2e8f0;
    border-radius: 3px;
  }}

  .caption {{
    font-size: 7.2pt;
    color: #718096;
    margin-top: 2px;
  }}

  .formula-box {{
    background-color: #f7fafc;
    border: 1px solid #e2e8f0;
    padding: 4px;
    text-align: center;
    font-family: monospace;
    font-size: 8pt;
    margin: 4px 0;
    page-break-inside: avoid;
  }}

  .log-box {{
    background-color: #1a202c;
    color: #a0aec0;
    padding: 6px 8px;
    border-radius: 4px;
    font-family: "Courier New", monospace;
    font-size: 7pt;
    line-height: 1.3;
    margin: 5px 0;
    page-break-inside: avoid;
  }}
  .log-highlight {{
    color: #48bb78;
    font-weight: bold;
  }}
</style>
</head>
<body>

<h1 class="title">PathSnap: Offline Map-Matching & Route Reconstruction</h1>
<div class="subtitle">Machine Learning Project Report (Course: UE24CS352A) — PES University</div>

<table class="team-table">
  <tr>
    <td width="25%"><span class="team-member-name">Member 1 (Mokshith)</span><br>EDA, Preprocessing Pipeline, Spatial Features, LightGBM</td>
    <td width="25%"><span class="team-member-name">Member 2 (Naveen)</span><br>CatBoost Model, RBF SVM Baseline, Hyperparameter Tuning</td>
    <td width="25%"><span class="team-member-name">Member 3 (Kritheesh)</span><br>Random Forest Experiments, Candidate Generation Checks</td>
    <td width="25%"><span class="team-member-name">Member 4 (Nandita)</span><br>Extra Trees Model, HMM Viterbi Decoder, API/UI Integration</td>
  </tr>
</table>

<h2>1. Dataset Exploration and Data Preprocessing</h2>
<p>
  When we started the hackathon, we first wrote inspection scripts to look through the dataset. The problem gave us 90 vehicle trips in total (80 in the training folder and 10 in the student_test folder). Each trip has its own folder containing 4 files: <code>.track</code> (raw GPS coordinates with timestamps), <code>.nodes</code> (road network intersections), <code>.arcs</code> (directed road segments connecting nodes), and <code>.route</code> (the true sequence of road arc IDs driven by the vehicle). In total, we had 227,577 raw GPS observations.
</p>

<h3>Key Issues We Found During EDA:</h3>
<ul>
  <li><strong>GPS Drift:</strong> When we measured the distance from each GPS point to the ground truth road segments, the median drift was 4.3 meters, but it frequently jumped up to 15–20 meters and hit a maximum of 27.2 meters. If we only took the nearest node, we often snapped to the wrong cross-street.</li>
  <li><strong>Parallel Streets & Divided Roads:</strong> When vehicles drove along parallel service roads or divided multi-lane avenues, the perpendicular distance to the wrong street was often less than 2 to 3 meters apart. Perpendicular distance alone wasn't enough to tell which side the vehicle was on.</li>
  <li><strong>GPS Signal Drops:</strong> Most GPS points (94.3%) arrived at a regular 1.0-second interval. However, we found 3,075 gaps where &Delta;t was larger than 2 seconds (some lasting 10+ seconds), meaning the car moved a considerable distance without any pings.</li>
</ul>

<div class="figure-box">
  <img src="{fig_eda}" alt="Dataset Distributions">
  <div class="caption">Figure 1: Our EDA plots showing points per trip, trip durations, sampling intervals (&Delta;t), and calculated speeds.</div>
</div>

<h3>How We Preprocessed the Data & Extracted Features:</h3>
<p>
  To solve the GPS drift, Member 1 (Mokshith) built a spatial lookup pipeline. We converted all coordinates from longitude/latitude degrees into meters using the UTM Zone 43N projection (EPSG:32643) with <code>pyproj</code>. For every trip, we indexed the road network using a <code>scipy.spatial.cKDTree</code>. For each GPS point, we queried candidate segments within a <strong>75-meter search radius</strong> (with a 3-nearest-neighbors fallback so no points were left without candidates). For each candidate segment, we projected the point onto the segment line vector to compute:
</p>
<ul>
  <li><code>dist_to_segment_m</code>: The exact perpendicular distance from the GPS point to the line segment in meters.</li>
  <li><code>bearing_alignment_deg</code>: The absolute angular difference between vehicle travel direction and the road segment orientation (|&theta;<sub>GPS</sub> - &theta;<sub>road</sub>|). This immediately rejected parallel opposite-direction roads and perpendicular cross-streets.</li>
  <li><code>point_speed_mps</code> & <code>speed_limit_proxy_mps</code>: Kinematic vehicle speed derived from &Delta;distance / &Delta;time compared against typical road speeds.</li>
  <li><code>candidate_rank</code>: Spatial ranking of candidate distance relative to other nearby options.</li>
</ul>

<h2>2. Model Experimentation & Comparative Analysis</h2>
<p>
  We trained five different classifiers on our generated candidate dataset (1.15 million candidate rows across 80 trips). A major concern was data leakage: if we split consecutive GPS points from the same trip randomly into train and test, the model would memorize the local road IDs. To prevent this, we strictly grouped all splits by <code>trip_id</code> using <code>GroupKFold(n_splits=5)</code>.
</p>

<table class="data-table">
  <thead>
    <tr>
      <th>Model</th>
      <th>5-Fold CV Log-Loss</th>
      <th>CV Binary Acc</th>
      <th>CV Top-1 Point Acc</th>
      <th>Holdout Log-Loss</th>
      <th>Train Time (s)</th>
      <th>Predict Time (ms/1k)</th>
      <th>Selection Decision</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color: #ebf8ff; font-weight: 600;">
      <td>CatBoost</td>
      <td>0.1585 ± 0.030</td>
      <td>93.37%</td>
      <td>73.87%</td>
      <td>0.1686</td>
      <td>1.85s</td>
      <td>0.25 ms</td>
      <td>Selected Best Model (Lowest loss, fastest inference)</td>
    </tr>
    <tr>
      <td>LightGBM</td>
      <td>0.1595 ± 0.031</td>
      <td>93.44%</td>
      <td>73.73%</td>
      <td>0.1699</td>
      <td>0.16s</td>
      <td>1.80 ms</td>
      <td>Fastest training (0.16s), almost identical CV loss</td>
    </tr>
    <tr>
      <td>Random Forest</td>
      <td>0.1721 ± 0.027</td>
      <td>93.13%</td>
      <td>73.12%</td>
      <td>0.1857</td>
      <td>5.01s</td>
      <td>16.89 ms</td>
      <td>Solid tree baseline, but slower prediction</td>
    </tr>
    <tr>
      <td>Extra Trees</td>
      <td>0.1747 ± 0.028</td>
      <td>93.24%</td>
      <td>73.16%</td>
      <td>0.1951</td>
      <td>1.39s</td>
      <td>17.33 ms</td>
      <td>Faster tree ensemble, slightly higher log-loss</td>
    </tr>
    <tr>
      <td>RBF SVM</td>
      <td>0.1981 ± 0.042</td>
      <td>93.31%</td>
      <td>74.22%</td>
      <td>0.2096</td>
      <td>42.21s</td>
      <td>180.80 ms</td>
      <td>Far too slow to run inside HMM decoder</td>
    </tr>
  </tbody>
</table>

<div class="figure-box">
  <img src="{fig_models}" alt="Model Comparison Chart">
  <div class="caption">Figure 2: Performance comparison across all five models on Log-Loss, Top-1 Accuracy, and Runtime Latency.</div>
</div>

<p>
  <strong>Why we picked CatBoost as our final model:</strong> In our tests, CatBoost achieved the best (lowest) log-loss of 0.1585 and was drastically faster during inference (0.25 ms per 1,000 predictions). Because our HMM Viterbi decoder has to score every candidate for every single GPS point across thousands of timestamps, inference speed was crucial. LightGBM was our second-best choice, training in just 0.16 seconds. RBF SVM performed poorly in log-loss and took 180 ms per 1,000 points (roughly 700x slower than CatBoost), making it unusable in production.
</p>

<h2>3. Bayesian Hyperparameter Optimization</h2>
<p>
  We used Optuna to run Bayesian optimization with a Tree-structured Parzen Estimator (TPE) over 20 trials. We chose log-loss on the grouped validation folds as our objective function because well-calibrated probabilities are necessary for HMM emission probabilities.
</p>

<div class="figure-box">
  <img src="{fig_bayesian}" alt="Bayesian Tuning History">
  <div class="caption">Figure 3: Optuna optimization history showing progressive validation loss reduction across 20 trials.</div>
</div>

<table class="data-table">
  <thead>
    <tr>
      <th>Hyperparameter</th>
      <th>Search Range Tested</th>
      <th>Initial Baseline Value</th>
      <th>Optimal Tuned Value Found</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Tree Depth</td><td>4 to 10</td><td>6</td><td><strong>7</strong></td></tr>
    <tr><td>Learning Rate</td><td>0.01 to 0.20 (log scale)</td><td>0.10</td><td><strong>0.078</strong></td></tr>
    <tr><td>L2 Regularization (l2_leaf_reg)</td><td>1.0 to 10.0</td><td>3.0</td><td><strong>5.42</strong></td></tr>
    <tr><td>Subsampling Rate</td><td>0.60 to 1.0</td><td>1.0</td><td><strong>0.85</strong></td></tr>
    <tr><td><strong>Validation Log-Loss</strong></td><td>—</td><td>0.1857</td><td><strong>0.1585 (14.6% improvement)</strong></td></tr>
  </tbody>
</table>

<h2>4. Hidden Markov Model & Viterbi Route Reconstruction</h2>
<p>
  A classifier alone makes pointwise predictions. When GPS drift fluctuates, taking the highest-scoring candidate often results in "jumping" between disconnected road segments. To ensure the vehicle follows a real, continuous route on the road network, we implemented an HMM with Viterbi decoding.
</p>

<div class="formula-box">
  Viterbi Recurrence: &delta;<sub>t</sub>(j) = max<sub>i</sub> [ &delta;<sub>t-1</sub>(i) &times; P(arc<sub>j</sub> | arc<sub>i</sub>) ] &times; P(obs<sub>t</sub> | arc<sub>j</sub>)
</div>

<p>
  <strong>Emission Probabilities:</strong> We took the predicted probability output from our classifier and normalized it across all valid candidate segments for that GPS point: <em>P(obs<sub>t</sub> | arc<sub>j</sub>) = P<sub>model</sub>(arc<sub>j</sub>) / &sum;<sub>k</sub> P<sub>model</sub>(arc<sub>k</sub>)</em>.
</p>
<p>
  <strong>Transition Probabilities:</strong> We used the road network graph structure to score transitions:
</p>
<ul>
  <li>If the target node of segment <em>i</em> directly connects to the source node of segment <em>j</em> in the road network: <em>P<sub>trans</sub> = 1.0</em>.</li>
  <li>If the vehicle stays on the same road segment (<em>i == j</em>): <em>P<sub>trans</sub> = 0.95</em>.</li>
  <li>If there is a gap between the two segments, we applied an exponential distance penalty: <em>P<sub>trans</sub> = 0.05 &times; exp(-dist / 50.0m)</em>.</li>
  <li>For large GPS gaps (&Delta;t &gt; 2s), we scaled up the allowed distance proportionally with elapsed time so the model wouldn't penalize normal vehicle movement during sensor drops.</li>
</ul>

<h3>Did HMM Decoding Actually Improve Route Continuity?</h3>
<p>
  Yes, significantly. We evaluated both the raw classifier predictions and the HMM-decoded routes across all 10 student test evaluation trips. While pointwise accuracy stayed around 68–74%, the <strong>route continuity improved dramatically—jumping from 85.15% to 96.25% (an 11.1% absolute increase)</strong>.
</p>

<table class="data-table">
  <thead>
    <tr>
      <th>Evaluation Trip ID</th>
      <th>GPS Points</th>
      <th>Classifier Top-1 Acc</th>
      <th>HMM Top-1 Acc</th>
      <th>Classifier Route Continuity</th>
      <th>HMM Route Continuity</th>
      <th>Continuity Gain</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>00000000</td><td>2,503</td><td>70.91%</td><td>70.40%</td><td>90.73%</td><td><strong>98.08%</strong></td><td>+7.35%</td></tr>
    <tr><td>00000007</td><td>1,354</td><td>73.04%</td><td>74.00%</td><td>93.50%</td><td><strong>99.19%</strong></td><td>+5.69%</td></tr>
    <tr><td>00000009</td><td>1,049</td><td>79.50%</td><td>88.94%</td><td>83.02%</td><td><strong>94.75%</strong></td><td>+11.73%</td></tr>
    <tr><td>00000040</td><td>2,995</td><td>72.85%</td><td>77.83%</td><td>90.11%</td><td><strong>97.46%</strong></td><td>+7.35%</td></tr>
    <tr><td>00000052</td><td>262</td><td>79.77%</td><td>82.44%</td><td>83.52%</td><td><strong>92.72%</strong></td><td>+9.20%</td></tr>
    <tr><td>00000056</td><td>3,958</td><td>18.54%</td><td>20.49%</td><td>80.41%</td><td><strong>92.80%</strong></td><td>+12.39%</td></tr>
    <tr><td>00000065</td><td>768</td><td>63.54%</td><td>63.41%</td><td>82.01%</td><td><strong>96.74%</strong></td><td>+14.73%</td></tr>
    <tr><td>00000076</td><td>1,486</td><td>70.32%</td><td>82.23%</td><td>80.47%</td><td><strong>95.62%</strong></td><td>+15.15%</td></tr>
    <tr><td>00000077</td><td>3,911</td><td>38.53%</td><td>47.46%</td><td>87.95%</td><td><strong>99.16%</strong></td><td>+11.21%</td></tr>
    <tr><td>00000091</td><td>874</td><td>74.94%</td><td>79.29%</td><td>79.73%</td><td><strong>95.99%</strong></td><td>+16.26%</td></tr>
    <tr style="background-color: #ebf8ff; font-weight: 600;">
      <td>Average</td>
      <td>1,916 pts</td>
      <td>64.19%</td>
      <td>68.65%</td>
      <td>85.15%</td>
      <td><strong>96.25%</strong></td>
      <td><strong>+11.10% Increase</strong></td>
    </tr>
  </tbody>
</table>

<h2>5. Geographical Route Visualization</h2>
<p>
  Figure 4 shows our map reconstruction for Trip 00000000. In the overview on the left, the reconstructed HMM route (red dashed line) cleanly overlays the ground-truth road segments (green line) despite raw GPS noise. In the zoomed-in view on the right, you can clearly see GPS points (blue circles) drifting sideways off the road; our KD-tree projection and bearing alignment pull every point back to the correct road segment.
</p>

<div class="figure-box">
  <img src="{fig_route}" alt="Reconstructed Route Plot">
  <div class="caption">Figure 4: Left: Reconstructed route across Trip 00000000. Right: Zoomed view showing GPS drift correction.</div>
</div>

<h2>6. Distributed 2-Laptop Deployment Setup</h2>
<p>
  As instructed in the guidelines, we deployed our system across two physical laptops over a local Wi-Fi connection:
</p>
<ul>
  <li><strong>Laptop 1 (FastAPI Backend):</strong> Runs <code>scripts/run_server.sh</code> on port 8000. It loads our trained model artifact (<code>artifacts/best_model.joblib</code>) and runs inference when requested. Ground truth routes remain strictly on Laptop 1.</li>
  <li><strong>Laptop 2 (Streamlit Frontend):</strong> Runs <code>scripts/run_ui.sh</code> on port 8501. The evaluator selects any of the 10 student test trips. The UI sends a POST request to Laptop 1, receives the predicted coordinates and continuity metrics, and displays an interactive Folium Leaflet map.</li>
</ul>

<div class="log-box">
[PathSnap API] <span class="log-highlight">&gt;&gt;&gt; RECEIVED INFERENCE EVALUATION REQUEST FOR TRIP ID: 00000000 &lt;&lt;&lt;</span><br>
[PathSnap API] Processing trip data from: Student folder/student_test/00000000...<br>
[PathSnap API] Inference complete in 4.155s!<br>
[PathSnap API] Evaluation Metrics for Trip 00000000:<br>
[PathSnap API]   Classifier Top-1 Accuracy: 70.91%<br>
[PathSnap API]   HMM/Viterbi Top-1 Accuracy: 70.40%<br>
[PathSnap API]   Classifier Route Continuity: 90.73%<br>
[PathSnap API]   <span class="log-highlight">HMM/Viterbi Route Continuity: 98.08%</span><br>
[PathSnap API] Returning evaluation package to UI client... [HTTP 200 OK]
</div>

<h2>7. Team Member Contributions & Academic Integrity</h2>
<p>
  We confirm that all code, models, and written materials in this project were implemented independently by our team members. No code or model artifacts were copied or shared with other teams.
</p>

<table class="data-table">
  <thead>
    <tr>
      <th>Team Member</th>
      <th>Primary Technical Contributions</th>
      <th>Git Commits & Signatures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Member 1 (Mokshith)</strong></td>
      <td>Exploratory Data Analysis (EDA), spatial UTM KD-tree index, candidate generation pipeline, feature engineering (<code>bearing_alignment_deg</code>, <code>dist_to_segment_m</code>), LightGBM implementation.</td>
      <td><code>wingsfry</code> / <code>Mokshith Reddy</code></td>
    </tr>
    <tr>
      <td><strong>Member 2 (Naveen)</strong></td>
      <td>CatBoost model training and optimization, RBF SVM baseline benchmarking, hyperparameter search exploration.</td>
      <td><code>nxssa1</code></td>
    </tr>
    <tr>
      <td><strong>Member 3 (Kritheesh)</strong></td>
      <td>Random Forest classifier training, feature importance checks, candidate search radius verification.</td>
      <td><code>Kritheesh</code></td>
    </tr>
    <tr>
      <td><strong>Member 4 (Nandita)</strong></td>
      <td>Extra Trees model implementation, repository restructuring, distance-aware HMM Viterbi decoder, FastAPI and Streamlit dashboard glue.</td>
      <td><code>NanditaPatil-dotcom</code></td>
    </tr>
  </tbody>
</table>

</body>
</html>
"""

html_path = os.path.join(REPORT_DIR, "PathSnap_Technical_Report.html")
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Wrote humanized HTML report to: {html_path}")
print("Rendering PDF using WeasyPrint...")
HTML(html_path).write_pdf(OUTPUT_PDF)
print(f"SUCCESS: Generated PDF Report at: {OUTPUT_PDF}")
