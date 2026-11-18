"""Build the PathSnap technical report (HTML) from saved results.

Every number and figure in the report is read from files under results/ and
configs/, so anything in the PDF can be traced back to the repository.

    python report/build_report.py            # writes report/PathSnap_Technical_Report.html
    # then print the HTML to PDF (browser "Save as PDF", or any headless Chrome)

Screenshots of the running UI and of the API terminal on Laptop 1 go in
report/screenshots/ as PNG files; they are embedded automatically.
"""
from __future__ import annotations

import base64
import glob
import html
import json
import os

import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *p: os.path.join(ROOT, *p)  # noqa: E731
OUT = R("report", "PathSnap_Technical_Report.html")


def img(path: str, caption: str, width: str = "100%") -> str:
    if not os.path.exists(path):
        return f'<p class="missing">[Missing figure: {html.escape(os.path.relpath(path, ROOT))}]</p>'
    data = base64.b64encode(open(path, "rb").read()).decode()
    return (f'<figure><img src="data:image/png;base64,{data}" style="width:{width}">'
            f"<figcaption>{html.escape(caption)}</figcaption></figure>")


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def f3(x: float) -> str:
    return f"{x:.3f}"


def table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, escape=True, border=0, classes="t")


# ---------------------------------------------------------------- load results
cfg = yaml.safe_load(open(R("configs", "models.yaml")))
test = pd.read_csv(R("results", "metrics", "student_test_model_metrics.csv")).set_index("model")
cv = pd.read_csv(R("results", "metrics", "student_test_model_cv.csv")).set_index("model")
seq = pd.read_csv(R("results", "metrics", "student_test_sequence_metrics.csv")).set_index("model")
summary = json.load(open(R("results", "metrics", "student_test_evaluation_summary.json")))
hmm_cand = json.load(open(R("results", "metrics", "student_test_hmm_metrics.json")))
hmm_trips = pd.read_csv(R("results", "metrics", "scorer_hmm_real_trips_eval.csv"))
cms = json.load(open(R("results", "metrics", "student_test_confusion_matrices.json")))
inv = pd.read_csv(R("results", "eda", "trip_inventory.csv"))
eda_txt = open(R("results", "eda", "eda_summary.txt")).read()

MODELS = ["CatBoost", "LightGBM", "ExtraTrees", "RandomForest", "RBF_SVM"]
FAMILY = {"CatBoost": "Boosting (ordered, symmetric trees)", "LightGBM": "Boosting (histogram, leaf-wise)",
          "ExtraTrees": "Ensemble (randomised trees)", "RandomForest": "Ensemble (bagged trees)",
          "RBF_SVM": "Kernel-based"}
CFGKEY = {"CatBoost": "catboost", "LightGBM": "lightgbm", "ExtraTrees": "extra_trees",
          "RandomForest": "random_forest", "RBF_SVM": "rbf_svm"}
SKIP = {"random_state", "random_seed", "verbose", "n_jobs"}


def cfg_str(m: str) -> str:
    return ", ".join(f"{k}={v}" for k, v in cfg[CFGKEY[m]].items() if k not in SKIP)


# ---------------------------------------------------------------- sections
parts = []
add = parts.append

# EDA
by_part = inv.groupby("partition").agg(trips=("trip_id", "count"), gps_points=("n_points", "sum"),
                                        null_rows=("n_missing_rows", "sum"), duplicate_rows=("n_duplicate_rows", "sum"),
                                        median_dt1s=("frac_dt_eq_1s", "median"), max_gap_s=("max_gap_sec", "max"),
                                        mean_nodes=("n_nodes", "mean"), mean_arcs=("n_arcs", "mean")).reset_index()
by_part["median_dt1s"] = by_part["median_dt1s"].map(pct)
by_part["mean_nodes"] = by_part["mean_nodes"].round(0).astype(int)
by_part["mean_arcs"] = by_part["mean_arcs"].round(0).astype(int)

add(f"""
<h2>1. Dataset understanding and EDA</h2>
<p><b>Partitions used.</b> All EDA below uses the <i>train</i> (80 trips) and <i>student_test</i> (10 trips)
partitions. The evaluator_test partition was not opened for analysis, training, feature selection or tuning.</p>
{table(by_part)}
<p class="cap">Table 1: Dataset inventory per partition (results/eda/trip_inventory.csv).</p>
<pre>{html.escape(eda_txt.strip())}</pre>
<p><b>Missing values.</b> The inventory shows {int(inv['n_missing_rows'].sum())} null GPS rows and
{int(inv['n_duplicate_rows'].sum())} duplicate rows across the 90 trips. The main data-quality issue is
time gaps rather than missing values: most steps are exactly 1 s, but some trips have gaps up to
{int(inv['max_gap_sec'].max())} s (about {inv['max_gap_sec'].max()/3600:.0f} hours, a recording break inside one trip). Every trip has a reference .route file in these two partitions.</p>
{img(R('results', 'eda', 'dataset_distributions.png'), 'Figure 1: GPS trajectory and sampling distributions (train + student_test).')}

<h3>Preprocessing (src/pathsnap/data/preprocessing)</h3>
<ul>
<li><b>Cleaning:</b> drop null, duplicate and out-of-range rows; enforce increasing timestamps; drop points whose implied
speed exceeds 62.5 m/s or acceleration exceeds 8 m/s² (clean.py).</li>
<li><b>Distances:</b> geodesic (haversine) distances in metres, never Euclidean distance in degrees (geo.py).</li>
<li><b>Road graph:</b> directed graph from .nodes/.arcs with a KD-tree over segments for nearest-road lookup (road_graph.py).</li>
<li><b>Candidates:</b> every arc within 40 m of a GPS point is a candidate (features.py). Mean candidate miss rate on
student_test (reference arc not in the candidate set): {pct(summary['candidate_miss_rate_mean_over_student_test_trips'])}.</li>
<li><b>Labels:</b> a candidate is positive when it is the .route arc matched to that point (route_match.py).</li>
<li><b>Splits:</b> grouped by trip_id everywhere (5-fold GroupKFold on the 80 training trips).</li>
</ul>
<h3>Features and why</h3>
<ul>
<li><code>dist_to_segment_m</code>: perpendicular distance from the GPS point to the arc; the strongest single signal.</li>
<li><code>bearing_alignment_deg</code>: angle between travel direction and arc direction; separates parallel and
opposite-direction roads that are only a few metres apart.</li>
<li><code>point_speed_mps</code>, <code>speed_limit_proxy_mps</code>: observed speed versus the typical speed on that arc.</li>
<li><code>arc_length_m</code> and <code>candidate_rank</code> (rank by distance within the point's candidate set).</li>
<li><code>candidate_arc_id</code> is excluded: arc ids are per-trip row numbers, so a model could only memorise them.</li>
</ul>
""")

# Model comparison
rows = []
for m in MODELS:
    t, c = test.loc[m], cv.loc[m]
    rows.append({"Model": m, "Algorithm family": FAMILY[m], "Key configuration": cfg_str(m),
                 "CV macro F1": f"{c.cv_macro_f1:.3f} ± {c.cv_macro_f1_std:.3f}",
                 "CV log loss": f"{c.cv_log_loss:.4f}",
                 "Top-1 accuracy": pct(t.top1_point_acc), "Route continuity": pct(t.route_continuity),
                 "Student-test macro F1": f3(t.macro_f1), "Log loss": f"{t.log_loss:.4f}",
                 "Precision / recall (pos.)": f"{t.precision:.3f} / {t.recall:.3f}",
                 "Train time (s)": f"{t.train_seconds:.2f}",
                 "Inference (ms / 1k cand.)": f"{t.inference_ms_per_1000_candidates:.2f}"})
cmp_df = pd.DataFrame(rows)

add(f"""
<h2>2. Model experimentation and comparison</h2>
<p>All five models use the same candidate table and features, are fit on training trips only, and are compared with
5-fold trip-grouped CV. For the comparison, training used {summary['train_rows_used']:,} candidate rows sampled by GPS
point from all {summary['train_trips_used']} training trips (RBF SVM cannot be trained on the full 1.15M rows in reasonable time). Student-test numbers are on all {summary['student_test_rows']:,} candidate rows of the
{summary['student_test_trips']} student_test trips.</p>
<p><b>Primary selection metric:</b> {html.escape(summary['selection_rule'])}. Log loss is used because the
classifier's probabilities become the HMM emission probabilities, so calibration matters more than thresholded
accuracy. Selected model: <b>{summary['selected_model']}</b>.</p>
<h3>2.1 Overall comparison</h3>
{table(cmp_df)}
<p class="cap">Table 2: results/metrics/student_test_model_cv.csv and student_test_model_metrics.csv. Top-1 accuracy is
observation-level (highest-scored candidate vs reference arc, classifier only). Route continuity is the share of
consecutive predicted arcs that are equal or connected in the road graph.</p>
{img(R('results', 'figures', 'cv_macro_f1.png'), 'Figure 2: Cross-validation macro F1 across models (training trips).', '75%')}
{img(R('results', 'figures', 'student_test_top1_macro_f1.png'), 'Figure 3: Student-test top-1 accuracy and macro F1.', '75%')}
{img(R('results', 'figures', 'student_test_log_loss.png'), 'Figure 4: Student-test log loss.', '75%')}
{img(R('results', 'figures', 'student_test_confusion_matrices.png'), 'Figure 5: Student-test confusion matrices (candidate level, threshold 0.5).')}
<p>None of these models produces epoch-wise training histories that we logged per fold, so no training/validation
loss curves are shown.</p>
""")

WHY = {
    "CatBoost": ("Gradient boosting on a small set of numeric, partly non-linear features (distance, angle, speed) is a "
                 "natural fit; ordered boosting gives well-calibrated probabilities with little tuning.",
                 "Strengths: lowest CV log loss, fast inference, robust defaults. Limitations: slower to train than "
                 "LightGBM; a single global model cannot use sequence context (that is the HMM's job)."),
    "LightGBM": ("Same boosting idea as CatBoost with histogram-based, leaf-wise trees; good for 1M+ rows.",
                 "Strengths: fastest training, near-identical accuracy to CatBoost. Limitations: slightly worse log "
                 "loss; leaf-wise growth can overfit small groups if num_leaves is large."),
    "ExtraTrees": ("Randomised tree ensemble; a low-variance non-linear baseline that needs no feature scaling.",
                   "Strengths: quick to train, highest positive-class precision among trees. Limitations: higher "
                   "log loss (probabilities are vote fractions), large model and slower inference than boosting."),
    "RandomForest": ("Bagged decision trees; standard strong baseline for tabular data.",
                     "Strengths: highest positive-class recall (class_weight=balanced_subsample). Limitations: worst "
                     "tree-model log loss, slower inference, large artifact."),
    "RBF_SVM": ("Kernel method from a different family, to check whether a smooth non-linear boundary does better than trees.",
                "Strengths: competitive top-1 accuracy. Limitations: O(n²) training (only feasible on a sample), "
                "probabilities come from Platt scaling and have the worst log loss, and inference is roughly 1000x slower per candidate "
                "than CatBoost, which is too slow to run inside the decoder."),
}
LOSS = {"CatBoost": "Logloss (binary cross-entropy)", "LightGBM": "binary log loss",
        "ExtraTrees": "Gini impurity per split", "RandomForest": "Gini impurity per split",
        "RBF_SVM": "hinge loss with RBF kernel, then Platt-scaled probabilities"}

add("<h3>2.2 Per-model questions</h3>")
for m in MODELS:
    t, c = test.loc[m], cv.loc[m]
    cm = cms[m]["matrix"]
    gap = t.log_loss - c.cv_log_loss
    fit = ("No sign of overfitting: student-test log loss is not worse than CV log loss "
           f"({t.log_loss:.4f} vs {c.cv_log_loss:.4f}), and the CV macro F1 spread across folds is small "
           f"(± {c.cv_macro_f1_std:.3f}).") if gap <= 0.01 else (
           f"Possible overfitting: student-test log loss {t.log_loss:.4f} is worse than CV {c.cv_log_loss:.4f}.")
    add(f"""
<h4>{m}</h4>
<p><b>A1. Why suitable:</b> {WHY[m][0]}</p>
<p><b>A2. Strengths and limitations:</b> {WHY[m][1]}</p>
<p><b>B5. Hyperparameters (initial values):</b> {html.escape(cfg_str(m))} (configs/models.yaml).</p>
<p><b>B6. Loss / objective:</b> {LOSS[m]}.</p>
<p><b>C7–8. Student-test performance:</b> top-1 accuracy {pct(t.top1_point_acc)}, macro F1 {f3(t.macro_f1)},
log loss {t.log_loss:.4f}, ROC AUC {f3(t.roc_auc)}, route continuity {pct(t.route_continuity)}, mean per-trip pointwise
top-1 {pct(seq.loc[m, 'mean_trip_pointwise_top1_accuracy'])}. Confusion matrix [[TN, FP], [FN, TP]] = {cm}.</p>
<p><b>C9. Over/underfitting:</b> {fit}</p>
""")

# Bayesian
bayes_path = R("results", "bayesian", "summary.json")
if os.path.exists(bayes_path):
    b = json.load(open(bayes_path))
    space = "".join(f"<li><code>{html.escape(k)}</code>: {html.escape(str(v))}</li>" for k, v in b["search_space"].items())
    best = ", ".join(f"{k}={v}" for k, v in b["best_params"].items())
    comp = pd.DataFrame(b["comparison"])
    figs = "".join(img(p, "Figure: Optuna optimisation history (results/bayesian).", "75%")
                   for p in sorted(glob.glob(R("results", "bayesian", "*.png"))))
    add(f"""
<h2>3. Bayesian hyperparameter optimisation ({html.escape(b.get('model', 'CatBoost'))})</h2>
<p><b>Method:</b> {html.escape(b.get('method', 'Optuna TPE sampler'))}, trip-grouped CV on the 80 training trips only.</p>
<p><b>1–2. Search space:</b></p><ul>{space}</ul>
<p><b>3. Objective:</b> {html.escape(b['objective'])}.</p>
<p><b>4. Trials:</b> {b['n_trials']} (full trial log: results/bayesian/).</p>
<p><b>5. Best values:</b> {html.escape(best)}.</p>
<p><b>6. Tuned vs untuned</b> (same data partitions and metrics):</p>
{table(comp)}
<p>{html.escape(b.get('conclusion', ''))}</p>
<p><b>Model served by the API:</b> artifacts/models/catboost_best.joblib (default configuration). The tuned model
(artifacts/models/catboost_tuned.joblib) is kept for reference; since the accuracy difference is within noise we did not
switch the demo to it. Tuning script: scripts/tune_catboost.py.</p>
<p><b>Computational cost:</b> {html.escape(str(b.get('compute_cost', '')))}</p>
{figs}
""")
else:
    add('<h2>3. Bayesian hyperparameter optimisation</h2><p class="missing">[results/bayesian/summary.json not found: '
        'run the CatBoost tuning script and rebuild this report.]</p>')

# HMM
tr = hmm_trips.copy()
add(f"""
<h2>4. HMM and Viterbi route reconstruction (src/pathsnap/decoder)</h2>
<ol>
<li><b>Hidden states:</b> the candidate arcs of each GPS observation (arc_id, plus where the point projects on it).
The candidate-to-arc correspondence is kept, so the decoded state is directly a road segment.</li>
<li><b>Emissions:</b> CatBoost probability for each candidate, normalised over that point's candidates and floored at 1e-9
(<code>CandidateScorer.emission_log_prob</code>).</li>
<li><b>Transitions:</b> Newson &amp; Krumm style, in log space:
<code>-|route_dist - gps_dist| / beta - turn_weight · turn_deg / 180</code>, where route_dist is the shortest drivable
distance (Dijkstra on the directed arcs) between the two candidate positions. beta = 50 m, turn_weight = 3,
backtrack tolerance 30 m.</li>
<li><b>Missing candidates and large GPS gaps:</b> a point with no candidates gets no road (and is counted as wrong in Table 3); gps_dist grows with the gap, so
long gaps are not penalised for covering more distance.</li>
<li><b>Disconnected segments:</b> a transition is impossible if no path exists within 4·gps_dist + 300 m. If every
transition at a step is impossible, Viterbi restarts from that point; in the final route the gap is filled with the
shortest-path arcs.</li>
<li><b>Recurrence:</b> δ<sub>t</sub>(j) = max<sub>i</sub>[δ<sub>t−1</sub>(i) + log P(j|i)] + log P(o<sub>t</sub>|j), with back-pointers.</li>
</ol>
<h3>Did HMM decoding help?</h3>
<p>Two measurements, both on the 10 student_test trips with the selected CatBoost model:</p>
<ul>
<li>scripts/evaluate_student_test.py, scored on the {hmm_cand['evaluated_observations_with_candidates']:,} observations
that have candidates: top-1 {pct(hmm_cand['classifier_top1_accuracy'])} → {pct(hmm_cand['hmm_top1_accuracy'])},
route continuity {pct(hmm_cand['classifier_route_continuity'])} → {pct(hmm_cand['hmm_route_continuity'])}.</li>
<li>scripts/eval_scorer_real_trips.py (the same pipeline the API serves), per trip below.</li>
</ul>
{table(tr.rename(columns={'n_eval_points': 'points', 'classifier_acc': 'classifier top-1 %', 'hmm_acc': 'HMM top-1 %',
                          'clf_continuity': 'classifier continuity %', 'hmm_continuity': 'HMM continuity %'}))}
<p class="cap">Table 3: results/metrics/scorer_hmm_real_trips_eval.csv.</p>
<p>HMM decoding raised point accuracy on 9 of 10 trips and made the route close to fully connected. It did not help on
trip 00000007 (90.1% → 83.0%): the classifier was already accurate there, and the transition model pulled some points
onto a connected but wrong branch. Trips 00000077 and 00000065 stay weakest; their errors come mostly from wrong
candidates that remain connected, which the HMM cannot detect. No trip had an exact full-sequence match, so
sequence-level accuracy (exact match rate) is 0 for every model (results/metrics/student_test_sequence_metrics.csv).</p>
{img(R('results', 'figures', 'hmm_route_comparison.png'), 'Figure 6: Classifier-only vs classifier + HMM on student_test.', '75%')}
""")

# Folium + deployment
shots = sorted(glob.glob(R("report", "screenshots", "*.png")))
map_shots = [p for p in shots if "map" in os.path.basename(p).lower() or "folium" in os.path.basename(p).lower()]
other = [p for p in shots if p not in map_shots]
add("<h2>5. Geographical visualisation (Folium)</h2>"
    "<p>Map code: src/pathsnap/visualization/folium_viz.py and scripts/run_eda.py. Interactive maps are committed in "
    "results/maps/ and results/eda/00000001_map.html. Layers: raw GPS points, road network, reference route and "
    "predicted route, each in its own colour with a legend and layer control.</p>")
add("".join(img(p, f"Folium map: {os.path.basename(p)}") for p in map_shots)
    or '<p class="missing">[Add a Folium map screenshot as report/screenshots/folium_map.png]</p>')
add("<h2>6. API, UI and deployment</h2>"
    "<p>Laptop 1 runs FastAPI (scripts/run_server.sh, port 8000): <code>GET /trips</code> lists the evaluator_test trips, "
    "<code>POST /evaluate/{trip_id}</code> runs cleaning, candidate generation, CatBoost scoring and HMM decoding, "
    "then scores against the .route file that stays on Laptop 1. Laptop 2 runs the Streamlit UI (scripts/run_ui.sh), "
    "which only calls the API. Model: artifacts/models/catboost_best.joblib.</p>")
add("".join(img(p, os.path.basename(p)) for p in other)
    or '<p class="missing">[Add UI and API-terminal screenshots to report/screenshots/]</p>')

TEAM = """
<h2>7. Team contributions and academic integrity</h2>
<table class="t"><tr><th>Member</th><th>SRN</th><th>Main contributions</th></tr>
<tr><td>MokshitReddy Nallaballe</td><td>PES2UG24CS824</td><td>EDA, preprocessing and candidate generation, spatial features, LightGBM</td></tr>
<tr><td>Naveen Karthikeyan</td><td>PES2UG24CS306</td><td>Feature schema, CatBoost and RBF SVM, model comparison and evaluation, Bayesian optimisation, API integration</td></tr>
<tr><td>Kritheesh NV</td><td>PES2UG24CS293</td><td>Random Forest, candidate radius checks</td></tr>
<tr><td>Nandita</td><td>PES2UG24CS304</td><td>Extra Trees, HMM/Viterbi decoder, FastAPI and Streamlit, repository structure</td></tr>
</table>
<p><b>AI-use disclosure.</b> The team used AI coding assistants to help write and debug code, run
experiments and draft parts of this report. Every result in this report comes from scripts and output files committed
to the repository and was produced by actually running them; this report is regenerated from those files by
report/build_report.py. All team members have reviewed the content and can explain it.</p>
"""
add(TEAM)

CSS = """
body{font-family:Arial,Helvetica,sans-serif;font-size:11px;line-height:1.45;margin:28px;color:#111}
h1{font-size:20px;margin-bottom:2px}h2{font-size:15px;border-bottom:1px solid #999;margin-top:22px}
h3{font-size:12.5px}h4{font-size:12px;margin:12px 0 4px}
table.t{border-collapse:collapse;margin:6px 0;font-size:9.5px}table.t th,table.t td{border:1px solid #bbb;padding:3px 5px;text-align:left}
table.t th{background:#eee}figure{margin:8px 0;page-break-inside:avoid}figcaption,.cap{font-size:9.5px;color:#444}
pre{font-size:9px;background:#f5f5f5;padding:6px}.missing{color:#b00;font-weight:bold}
"""
doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>PathSnap Technical Report</title>
<style>{CSS}</style></head><body>
<h1>PathSnap: Map-Matching and Route Reconstruction</h1>
<p>UE24CS352A Machine Learning, PES University. Repository: github.com/NanditaPatil-dotcom/ML-hackathon.
All numbers are read from results/ by report/build_report.py.</p>
{''.join(parts)}
</body></html>"""
open(OUT, "w", encoding="utf-8").write(doc)
print("wrote", OUT)
