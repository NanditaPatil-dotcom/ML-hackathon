# ML-hackathon

PathSnap: GPS map-matching. This repo scores each GPS point's candidate road
segments (this section) and feeds those scores into an HMM/Viterbi step
(`hmm/`) and a FastAPI service (`api/`).

## Repo layout

```
configs/            data.yaml (mock-data + feature schema), models.yaml (hyperparams)
data/
  raw/               original trip data (not tracked in git)
  interim/           cleaned/converted intermediate files (not tracked)
  processed/         model-ready features + labels (e.g. candidates_mock.csv)
notebooks/           exploration.ipynb — scratch space, not required by the pipeline
src/pathsnap/
  data/              build_features.py — mock feature generator (stands in for
                      real preprocessing until it lands)
  models/            base.py (shared schema + model registry), catboost_model.py,
                      rbf_svm.py, scorer.py (CandidateScorer, the HMM-facing interface)
  evaluation/        metrics.py, compare_models.py — CV + holdout comparison
  tuning/            reserved for hyperparameter search (not yet implemented)
  decoder/           placeholder — HMM/Viterbi itself lives in hmm/ at repo root
  visualization/     reserved for map/plotting helpers
scripts/             train_all_models.py, evaluate_models.py, generate_mock_data.py,
                      smoke_test_scorer.py — the CLI entry points
artifacts/models/    saved model files (CandidateScorer, joblib)
artifacts/logs/      training logs (not tracked)
results/metrics/     model_comparison.csv / .md, summary.json
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
same `configs/data.yaml` schema, so the full 5-model comparison the task spec
requires comes together from both.

Real preprocessing isn't ready yet, so this is built and tested against a
mocked dataset with the same column names the real pipeline will produce —
see `configs/data.yaml` for the exact schema
(`distance_to_segment, bearing_diff, speed_limit, transition_distance,
candidate_road_id, label`, plus `point_id`/`trip_id` for grouped splits).

```bash
pip install -r requirements.txt

# generate the mock dataset -> data/processed/candidates_mock.csv
python scripts/generate_mock_data.py

# train + compare -> results/metrics/model_comparison.{csv,md} and
# artifacts/models/<best_model>_best.joblib
python scripts/train_all_models.py --data data/processed/candidates_mock.csv

# once real preprocessed data lands, just point at it (same column names):
python scripts/train_all_models.py --data path/to/real_preprocessed.csv

# evaluate an already-trained model against a (new) dataset without retraining:
python scripts/evaluate_models.py --model artifacts/models/catboost_best.joblib --data data/processed/candidates_mock.csv
```

### Results on the mock dataset

| Model | CV log_loss | CV top1_point_acc | test predict_ms |
|---|---|---|---|
| RBF_SVM | 0.2223 ± 0.0125 | 0.858 | 60.9 |
| CatBoost | 0.2223 ± 0.0149 | 0.859 | 1.5 |

Statistically tied on log-loss; **CatBoost** wins on inference speed (needed
since the HMM scores every candidate of every GPS point) and is saved as
`artifacts/models/catboost_best.joblib`.

### Using the trained model (for the HMM / API)

```python
import sys; sys.path.insert(0, "src")
from pathsnap.models.scorer import CandidateScorer

scorer = CandidateScorer.load("artifacts/models/catboost_best.joblib")
df["p_match"] = scorer.predict_proba(df)        # P(candidate is the true road), in [0,1]
df["emission"] = scorer.emission_log_prob(df)   # log-prob, normalized per point_id
```
