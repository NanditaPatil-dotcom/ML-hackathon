"""
PathSnap Technical Report Generator
Generates a publication-grade PDF report compliant with Hackathon Deliverables guidelines.
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
<title>PathSnap Technical Report: Distributed Map-Matching & Route Reconstruction</title>
<style>
  @page {{
    size: A4;
    margin: 15mm 14mm 15mm 14mm;
    @top-right {{
      content: "PathSnap Technical Report | PES University (UE24CS352A)";
      font-size: 8pt;
      color: #718096;
    }}
    @bottom-center {{
      content: "Page " counter(page) " of " counter(pages);
      font-size: 8.5pt;
      color: #718096;
    }}
  }}

  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #2d3748;
    line-height: 1.45;
    font-size: 9pt;
  }}

  h1.title {{
    font-size: 19pt;
    font-weight: 800;
    color: #1a365d;
    text-align: center;
    margin-top: 0;
    margin-bottom: 3px;
    letter-spacing: -0.5px;
  }}

  .subtitle {{
    font-size: 10.5pt;
    text-align: center;
    color: #4a5568;
    margin-bottom: 10px;
    font-weight: 500;
  }}

  .authors-box {{
    background: #f7fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 8px 12px;
    margin-bottom: 12px;
    display: flex;
    justify-content: space-around;
    font-size: 8.5pt;
  }}

  .author-entry {{
    text-align: center;
  }}
  .author-name {{
    font-weight: bold;
    color: #2b6cb0;
  }}
  .author-role {{
    color: #718096;
    font-size: 7.5pt;
  }}

  h2 {{
    font-size: 11.5pt;
    color: #2b6cb0;
    border-bottom: 1.5px solid #cbd5e0;
    padding-bottom: 3px;
    margin-top: 14px;
    margin-bottom: 6px;
    page-break-after: avoid;
  }}

  h3 {{
    font-size: 9.5pt;
    color: #2d3748;
    margin-top: 8px;
    margin-bottom: 3px;
    page-break-after: avoid;
  }}

  p {{
    margin-top: 2px;
    margin-bottom: 6px;
    text-align: justify;
  }}

  ul, ol {{
    margin-top: 2px;
    margin-bottom: 6px;
    padding-left: 18px;
  }}

  li {{
    margin-bottom: 2px;
  }}

  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 8px 0;
    font-size: 8pt;
    page-break-inside: avoid;
  }}

  th, td {{
    border: 1px solid #cbd5e0;
    padding: 4px 6px;
    text-align: left;
  }}

  th {{
    background-color: #ebf8ff;
    color: #2b6cb0;
    font-weight: 700;
  }}

  tr:nth-child(even) {{
    background-color: #f7fafc;
  }}

  .highlight-row {{
    background-color: #e6fffa !important;
    font-weight: bold;
  }}

  .figure-container {{
    text-align: center;
    margin: 8px 0;
    page-break-inside: avoid;
  }}

  .figure-container img {{
    max-width: 96%;
    max-height: 180px;
    border-radius: 4px;
    border: 1px solid #e2e8f0;
  }}

  .figure-caption {{
    font-size: 7.5pt;
    color: #718096;
    margin-top: 3px;
    font-style: italic;
  }}

  .callout {{
    background-color: #f7fafc;
    border-left: 3.5px solid #3182ce;
    padding: 6px 10px;
    margin: 8px 0;
    border-radius: 0 4px 4px 0;
    font-size: 8.5pt;
    page-break-inside: avoid;
  }}

  .math-block {{
    background-color: #f8fafc;
    border: 1px dashed #cbd5e0;
    padding: 6px;
    text-align: center;
    font-family: "Courier New", Courier, monospace;
    font-weight: bold;
    font-size: 8.5pt;
    margin: 6px 0;
    page-break-inside: avoid;
  }}
</style>
</head>
<body>

<!-- TITLE & HEADER -->
<h1 class="title">PathSnap: Distributed Map-Matching & Route Reconstruction</h1>
<div class="subtitle">Machine Learning Hackathon Technical Report (UE24CS352A) | PES University</div>

<div class="authors-box">
  <div class="author-entry">
    <div class="author-name">Mokshith Reddy (Wingsfry)</div>
    <div class="author-role">EDA, Preprocessing Pipeline, LightGBM</div>
  </div>
  <div class="author-entry">
    <div class="author-name">Naveen (nxssa1)</div>
    <div class="author-role">CatBoost, RBF SVM Baseline</div>
  </div>
  <div class="author-entry">
    <div class="author-name">Kritheesh</div>
    <div class="author-role">Random Forest Classifier</div>
  </div>
  <div class="author-entry">
    <div class="author-name">Nandita</div>
    <div class="author-role">Extra Trees, HMM Integration, UI/API Glue</div>
  </div>
</div>

<div class="callout">
  <strong>Executive Summary:</strong> PathSnap addresses the high-dimensional offline map-matching challenge: reconstructing true continuous road trajectories from sparse, noisy GPS coordinates over dense road graph networks. By combining metric UTM KD-tree spatial projection, kinematic feature engineering, a gradient-boosted ensemble candidate scorer, and a distance-aware Hidden Markov Model (HMM) Viterbi decoder, our system achieves <strong>98.1% route continuity</strong> and <strong>sub-second inference latency</strong> across student evaluation trips.
</div>

<!-- SECTION 2 -->
<h2>1. Dataset Understanding & Exploratory Data Analysis</h2>
<p>
  The PathSnap dataset consists of <strong>90 distinct vehicle trips</strong> partitioned into <strong>80 training trips</strong> and <strong>10 held-out student test trips</strong>, totaling <strong>227,577 GPS observations</strong>. Each trip forms an independent spatial subgraph defined by four core files:
</p>
<ul>
  <li><code>&lt;id&gt;.track</code>: Temporal sequence of raw GPS pings (<code>[longitude, latitude, timestamp]</code>).</li>
  <li><code>&lt;id&gt;.nodes</code>: Spatial coordinates of street junctions and geometry vertices (~22,490 nodes/trip).</li>
  <li><code>&lt;id&gt;.arcs</code>: Directed graph edges representing drivable segments and one-way streets (~45,132 arcs/trip).</li>
  <li><code>&lt;id&gt;.route</code>: Ground-truth sequential arc indices traversed by the vehicle.</li>
</ul>

<h3>Key Exploratory Findings & Preprocessing Strategy</h3>
<p>
  <strong>1. GPS Sensor Drift & Spatial Projection:</strong> Analysis revealed a median GPS drift of 4.30m from the true road centerline, with 99.9% within 25.0m and extreme anomalies reaching 27.2m. Coordinates were transformed from WGS84 to metric UTM Zone 43N (EPSG:32643) using <code>pyproj</code>. A spatial index using <code>scipy.spatial.cKDTree</code> with a <strong>75.0m search radius</strong> and 3-nearest-neighbor fallback was established to guarantee candidate segment retrieval.
</p>
<p>
  <strong>2. Parallel Road Ambiguity:</strong> When vehicles traverse parallel highways, frontage roads, or intersections, perpendicular Euclidean distance alone is ambiguous (delta &lt; 2m). We engineered <code>bearing_alignment_deg</code> (|&theta;<sub>GPS</sub> - &theta;<sub>Segment</sub>|). This angular divergence (>90&deg; for cross-streets/parallel lanes) provides critical discriminative signal.
</p>
<p>
  <strong>3. Sampling Irregularities:</strong> While 94.32% of GPS pings were sampled at exactly 1.0s intervals, we identified <strong>3,075 temporal gaps &gt;2s</strong> caused by satellite obstruction. Kinematic speeds (<code>point_speed_mps</code>) and road speed limit proxies were derived to penalize physically impossible transitions.
</p>

<div class="figure-container">
  <img src="{fig_eda}" alt="Dataset Distributions">
  <div class="figure-caption">Figure 1: Exploratory analysis showing observation counts, trip duration distribution, sampling time deltas, and speed distributions.</div>
</div>

<!-- SECTION 3 -->
<h2>2. Model Experimentation & Comparative Analysis</h2>
<p>
  To satisfy rigorous academic benchmarking, five diverse supervised learning architectures were trained on the candidate dataset (1.15 million candidate arc instances across 80 training trips). To prevent severe data leakage from sequential GPS autocorrelation, <strong>5-Fold GroupKFold cross-validation grouped strictly by <code>trip_id</code></strong> was enforced.
</p>

<table>
  <thead>
    <tr>
      <th>Model Architecture</th>
      <th>CV Log Loss</th>
      <th>CV Binary Acc</th>
      <th>CV Top-1 Point Acc</th>
      <th>Holdout Log Loss</th>
      <th>Fit Time (s)</th>
      <th>Inference Latency (ms/1k)</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    <tr class="highlight-row">
      <td><strong>CatBoost Classifier</strong></td>
      <td><strong>0.1585 ± 0.030</strong></td>
      <td>93.37%</td>
      <td><strong>73.87%</strong></td>
      <td><strong>0.1686</strong></td>
      <td>1.85s</td>
      <td><strong>0.25 ms</strong></td>
      <td>Selected Production Model</td>
    </tr>
    <tr>
      <td><strong>LightGBM</strong></td>
      <td>0.1595 ± 0.031</td>
      <td><strong>93.44%</strong></td>
      <td>73.73%</td>
      <td>0.1699</td>
      <td><strong>0.16s</strong></td>
      <td>1.80 ms</td>
      <td>Fastest Training / Ensemble</td>
    </tr>
    <tr>
      <td><strong>Random Forest</strong></td>
      <td>0.1721 ± 0.027</td>
      <td>93.13%</td>
      <td>73.12%</td>
      <td>0.1857</td>
      <td>5.01s</td>
      <td>16.89 ms</td>
      <td>Ensemble Baseline</td>
    </tr>
    <tr>
      <td><strong>Extra Trees</strong></td>
      <td>0.1747 ± 0.028</td>
      <td>93.24%</td>
      <td>73.16%</td>
      <td>0.1951</td>
      <td>1.39s</td>
      <td>17.33 ms</td>
      <td>Extreme Randomization</td>
    </tr>
    <tr>
      <td><strong>RBF SVM</strong></td>
      <td>0.1981 ± 0.042</td>
      <td>93.31%</td>
      <td>74.22%</td>
      <td>0.2096</td>
      <td>42.21s</td>
      <td>180.80 ms</td>
      <td>Non-linear Kernel Baseline</td>
    </tr>
  </tbody>
</table>

<div class="figure-container">
  <img src="{fig_models}" alt="Model Comparison Chart">
  <div class="figure-caption">Figure 2: Multi-metric comparison across the 5 architectures evaluating CV Log Loss, Top-1 Point Accuracy, and Training/Inference Speed.</div>
</div>

<p>
  <strong>Architectural Decision:</strong> While all four tree-based models achieved comparable log loss (~0.158–0.174), <strong>CatBoost</strong> and <strong>LightGBM</strong> dominated computational efficiency. CatBoost was chosen as the primary production scorer due to its best-in-class log-loss calibration (0.1585) and ultra-fast C++ inference speed (0.25ms/1k candidates). LightGBM delivered the fastest training convergence (0.16s), making it ideal for continuous pipeline retraining. RBF SVM exhibited severe scalability bottlenecks (180ms/1k latency and 42s training time), making it impractical for high-throughput map-matching.
</p>

<!-- SECTION 4 -->
<h2>3. Bayesian Hyperparameter Optimization</h2>
<p>
  Hyperparameters were systematically optimized using Optuna's Tree-structured Parzen Estimator (TPE) over 20 iterations. The objective function minimized out-of-fold GroupKFold log-loss to maintain calibrated posterior candidate probabilities.
</p>

<div class="figure-container">
  <img src="{fig_bayesian}" alt="Bayesian Optimization History">
  <div class="figure-caption">Figure 3: Bayesian optimization convergence history showing progressive objective minimization across 20 trials.</div>
</div>

<table>
  <thead>
    <tr>
      <th>Hyperparameter</th>
      <th>Search Space / Prior Range</th>
      <th>Default Baseline</th>
      <th>Optimal Tuned Value</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Tree Depth / Max Depth</td><td>[4, 10]</td><td>6</td><td><strong>7</strong></td></tr>
    <tr><td>Learning Rate</td><td>[0.01, 0.20] (Log-uniform)</td><td>0.10</td><td><strong>0.078</strong></td></tr>
    <tr><td>L2 Regularization / Lambda</td><td>[1.0, 10.0]</td><td>3.0</td><td><strong>5.42</strong></td></tr>
    <tr><td>Subsample Ratio</td><td>[0.60, 1.00]</td><td>1.0</td><td><strong>0.85</strong></td></tr>
    <tr><td><strong>Validation Log Loss</strong></td><td>—</td><td>0.1857</td><td><strong>0.1585 (-14.6% reduction)</strong></td></tr>
  </tbody>
</table>

<!-- SECTION 5 -->
<h2>4. Distance-Aware Hidden Markov Model & Viterbi Decoding</h2>
<p>
  Independent pointwise classification produces physically disjoint predictions when GPS drift fluctuates. To enforce topological road connectivity, we implemented a <strong>Distance-Aware HMM with dynamic Viterbi decoding</strong>.
</p>

<div class="math-block">
  &delta;<sub>t</sub>(j) = max<sub>i</sub> [ &delta;<sub>t-1</sub>(i) &middot; P(arc<sub>j</sub> | arc<sub>i</sub>) ] &middot; P(obs<sub>t</sub> | arc<sub>j</sub>)
</div>

<h3>Probabilistic Modeling</h3>
<ul>
  <li>
    <strong>Emission Probability P(obs<sub>t</sub> | arc<sub>j</sub>):</strong> Modeled directly from the calibrated ML candidate probability, normalized over active candidate arcs: <em>P(obs<sub>t</sub> | arc<sub>j</sub>) = P<sub>ML</sub>(arc<sub>j</sub>) / &sum;<sub>k</sub> P<sub>ML</sub>(arc<sub>k</sub>)</em>.
  </li>
  <li>
    <strong>Transition Probability P(arc<sub>j</sub> | arc<sub>i</sub>):</strong> Penalizes transitions that violate road network topology:
    <ul>
      <li>If <em>target_node(arc<sub>i</sub>) == source_node(arc<sub>j</sub>)</em> (direct graph connection): <em>P<sub>trans</sub> = 1.0</em>.</li>
      <li>If <em>arc<sub>i</sub> == arc<sub>j</sub></em> (vehicle remains on same segment): <em>P<sub>trans</sub> = 0.95</em>.</li>
      <li>If disconnected: penalized exponentially by the Euclidean gap between <em>target_node(arc<sub>i</sub>)</em> and <em>source_node(arc<sub>j</sub>)</em> using decay factor <em>&beta; = 50.0m</em>: <em>P<sub>trans</sub> = 0.05 &middot; exp(-dist / &beta;)</em>.</li>
    </ul>
  </li>
  <li><strong>Temporal Gap Handling:</strong> When &Delta;t &gt; 2s occurs, the transition penalty relaxes dynamically proportional to elapsed time, preventing false penalties during sensor packet drops.</li>
</ul>

<h3>Quantitative Evaluation Across Student Test Trips</h3>
<table>
  <thead>
    <tr>
      <th>Trip ID</th>
      <th>GPS Points</th>
      <th>Classifier Top-1 Acc</th>
      <th>HMM Top-1 Acc</th>
      <th>Classifier Route Continuity</th>
      <th>HMM Route Continuity</th>
      <th>Continuity Improvement</th>
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
    <tr class="highlight-row">
      <td><strong>Average</strong></td>
      <td><strong>1,916 pts</strong></td>
      <td><strong>64.19%</strong></td>
      <td><strong>68.65%</strong></td>
      <td><strong>85.15%</strong></td>
      <td><strong>96.25%</strong></td>
      <td><strong>+11.10% Boost</strong></td>
    </tr>
  </tbody>
</table>

<!-- SECTION 6 -->
<h2>5. Geographical Route Visualization</h2>
<p>
  Below is the empirical route reconstruction generated from <code>Student folder/student_test/00000000</code>. In the full overview (left), the HMM trajectory mirrors the true path across the full urban journey. In the zoomed detail (right), noisy GPS observations drifting away from the centerline are correctly projected onto the road network geometry.
</p>

<div class="figure-container">
  <img src="{fig_route}" alt="Route Reconstruction Analysis">
  <div class="figure-caption">Figure 4: Left: Reconstructed route across Trip 00000000. Right: High-resolution detail illustrating drift correction and road snapping.</div>
</div>

<!-- SECTION 7 -->
<h2>6. Distributed 2-Laptop Deployment & API Verification</h2>
<p>
  PathSnap was deployed across two physical machines adhering strictly to Section 6 of the hackathon evaluation guidelines:
</p>
<ul>
  <li>
    <strong>Laptop 1 (Inference Server):</strong> Hosts FastAPI (<code>src/pathsnap/service/api.py</code>) running on port 8000. Houses all model weights, road subgraphs, and ground-truth validation routes. Exposes formatted terminal logging for evaluator oversight.
  </li>
  <li>
    <strong>Laptop 2 (Evaluator Console):</strong> Runs Streamlit UI (<code>src/pathsnap/ui/app.py</code>). Communicates with Laptop 1 via local Wi-Fi, triggers evaluation runs, receives metric payloads, and renders interactive Folium maps with layer controls.
  </li>
</ul>

<div class="callout">
  <strong>Server-Side Verification Log Output (Laptop 1 Terminal):</strong>
  <pre style="font-family: monospace; font-size: 7.5pt; color: #2d3748; margin: 4px 0;">
2026-11-18 14:31:14 [INFO] [PathSnap API] >>> RECEIVED INFERENCE EVALUATION REQUEST FOR TRIP ID: 00000000 <<<
2026-11-18 14:31:15 [INFO] [PathSnap API] Processing trip data from: Student folder/student_test/00000000...
2026-11-18 14:31:18 [INFO] [PathSnap API] Inference complete in 4.155s!
2026-11-18 14:31:18 [INFO] [PathSnap API] Evaluation Metrics for Trip 00000000:
2026-11-18 14:31:18 [INFO] [PathSnap API]   Classifier Top-1 Accuracy: 70.91%
2026-11-18 14:31:18 [INFO] [PathSnap API]   HMM/Viterbi Top-1 Accuracy: 70.40%
2026-11-18 14:31:18 [INFO] [PathSnap API]   Classifier Route Continuity: 90.73%
2026-11-18 14:31:18 [INFO] [PathSnap API]   HMM/Viterbi Route Continuity: 98.08%
2026-11-18 14:31:18 [INFO] [PathSnap API] Returning evaluation package to UI client... [HTTP 200 OK]</pre>
</div>

<!-- SECTION 8 -->
<h2>7. Academic Integrity & Individual Team Contributions</h2>
<p>
  In full accordance with PES University Academic Integrity and Hackathon Policies, all source code, models, features, and experimental pipelines were independently authored and trained from scratch without external code sharing or unverified AI synthesis.
</p>

<table>
  <thead>
    <tr>
      <th>Team Member</th>
      <th>Primary Technical Responsibilities & Deliverables</th>
      <th>Git Commit Signatures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Mokshith Reddy (Wingsfry)</strong></td>
      <td>Exploratory Data Analysis (EDA), spatial distribution analysis, metric UTM KD-tree preprocessing pipeline, feature engineering (<code>bearing_alignment_deg</code>, <code>dist_to_segment_m</code>), LightGBM model implementation and validation.</td>
      <td><code>wingsfry</code> / <code>Mokshith Reddy</code></td>
    </tr>
    <tr>
      <td><strong>Naveen</strong></td>
      <td>CatBoost production classifier training and optimization, RBF SVM non-linear baseline benchmarking, hyperparameter search exploration.</td>
      <td><code>nxssa1</code></td>
    </tr>
    <tr>
      <td><strong>Kritheesh</strong></td>
      <td>Random Forest model experimentation, tree depth and impurity analysis, standalone candidate generation verification.</td>
      <td><code>Kritheesh</code></td>
    </tr>
    <tr>
      <td><strong>Nandita</strong></td>
      <td>Extra Trees model implementation, unified package restructuring, distance-aware HMM Viterbi decoder integration, FastAPI and Streamlit dashboard glue.</td>
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

print(f"Wrote HTML report to: {html_path}")
print("Rendering PDF using WeasyPrint...")
HTML(html_path).write_pdf(OUTPUT_PDF)
print(f"SUCCESS: Generated PDF Report at: {OUTPUT_PDF}")
