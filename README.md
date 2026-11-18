# ML-hackathon

PathSnap: GPS map-matching. This repo scores each GPS point's candidate road
segments (this section) and feeds those scores into an HMM/Viterbi step
(`hmm/`) and a FastAPI service (`api/`).

## Repo layout

```
configs/            data.yaml (feature schema + mock settings), models.yaml (hyperparams)
data/
  raw/               original trip data (not tracked in git)
  interim/           cleaned/converted intermediate files (not tracked)
  processed/         model-ready features + labels (e.g. candidates_mock.csv)
notebooks/           exploration.ipynb — scratch space, not required by the pipeline
src/pathsnap/
  data/              build_features.py — add_candidate_rank() + mock feature generator
  models/            base.py (shared schema + model registry), catboost_model.py,
                      rbf_svm.py, scorer.py (CandidateScorer, the HMM/API-facing interface)
  evaluation/        metrics.py, compare_models.py — CV + holdout comparison
  tuning/            reserved for hyperparameter search (not yet implemented)
  decoder/           placeholder — HMM/Viterbi itself lives in hmm/ at repo root
  visualization/     reserved for map/plotting helpers
scripts/             train_all_models.py, evaluate_models.py, generate_mock_data.py,
                      smoke_test_scorer.py — the CLI entry points
artifacts/models/    saved model files (CandidateScorer, joblib)
artifacts/logs/      training logs (not tracked)
results/metrics/     model_comparison.csv / .md, summary.json, student_test_eval.json
results/figures/     reserved for plots
results/maps/        reserved for matched-route visualizations
report/              final writeup (not owned by any thread yet)
api/, ui/            FastAPI service + UI (separate thread/PR)
hmm/                 HMM/Viterbi decoder (separate thread/PR)
```

## Candidate-segment model training (CatBoost + RBF SVM)

Trains and compares model families that score each GPS point's candidate road
segments. This part of the repo covers **CatBoost** and **RBF SVM**; teammates
are covering LightGBM, Extra Trees and Random Forest in parallel against the
same schema, so the full 5-model comparison the task spec requires comes
together from both.

Schema (`configs/data.yaml`) matches `data_processing/pipeline.py`'s real
candidates parquet column-for-column — no renaming needed between
preprocessing and modeling:

- `dist_to_segment_m`, `bearing_alignment_deg`, `speed_limit_proxy_mps`,
  `point_speed_mps`, `arc_length_m` — raw candidate features
- `candidate_rank` — added automatically (`add_candidate_rank`): rank by
  `dist_to_segment_m` within each GPS point's candidates, 0 = nearest.
  Substitute for `candidate_arc_id`, which is a **per-trip row index** (not a
  stable road id) and is deliberately **not** used as a model feature — using
  it as a categorical would overfit to the training trips and not generalize
  to evaluation trips.
- `is_ground_truth_arc` — label
- `trip_id`, `point_idx` — grouping for CV splits / top-1 accuracy

```bash
pip install -r requirements.txt

# generate the mock dataset (same schema; stands in when real data isn't at hand)
python scripts/generate_mock_data.py

# train + compare -> results/metrics/model_comparison.{csv,md} and
# artifacts/models/<best_model>_best.joblib
python scripts/train_all_models.py --data data_processing/outputs/train_candidates.parquet

# evaluate an already-trained model against a (new) dataset without retraining:
python scripts/evaluate_models.py --model artifacts/models/catboost_best.joblib --data data_processing/outputs/student_test_candidates.parquet
```

RBF SVM doesn't scale to the full real candidates table (roughly
O(n^2)-O(n^3) in rows), so the comparison itself runs on a `--max-points`
subsample (default 6000 GPS points, all candidates kept together); CatBoost
(gradient boosting, ~linear in rows) is then refit on the **full** dataset
and saved as the production model, so the shipped model isn't trained on a
subsample even though the comparison that picked it was.

### Results on the real dataset (80 train trips, 1.15M candidate rows)

5-fold grouped CV + 80/20 holdout, on a 6000-point subsample (RBF SVM's limit):

| Model | CV log_loss | CV top1_point_acc | test predict_ms (44.9k rows) |
|---|---|---|---|
| CatBoost | 0.1585 ± 0.0296 | 0.739 | 1.6 |
| RBF_SVM | 0.1981 ± 0.0420 | 0.742 | 959.2 |

**CatBoost** wins outright on log-loss (not just a tie broken by speed this
time) and is ~600x faster at inference. Refit on the full 1.15M-row training
set and saved as `artifacts/models/catboost_best.joblib`.

Held out on the 10 **student_test** trips (never used in training/tuning):

| accuracy | f1 | roc_auc | log_loss | top1_point_acc |
|---|---|---|---|---|
| 0.944 | 0.763 | 0.970 | 0.136 | 0.782 |

(Full comparison table: `results/metrics/model_comparison.md`. Earlier
mock-data numbers, from before real preprocessing landed, are superseded by
these.)

### Using the trained model (for the HMM / API)

```python
import sys; sys.path.insert(0, "src")
from pathsnap.data.build_features import add_candidate_rank
from pathsnap.models.scorer import CandidateScorer

scorer = CandidateScorer.load("artifacts/models/catboost_best.joblib")
df = add_candidate_rank(df)                     # needed if candidate_rank isn't already a column
df["p_match"] = scorer.predict_proba(df)         # P(candidate is the true road), in [0,1]
df["emission"] = scorer.emission_log_prob(df)    # log-prob, normalized per (trip_id, point_idx)
```
