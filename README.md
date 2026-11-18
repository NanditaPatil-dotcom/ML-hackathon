# PathSnap

PathSnap matches noisy GPS observations to road segments and reconstructs a
connected route. The repository is organized as one Python package under
`src/pathsnap`; command-line entry points live in `scripts/`.

## Repository layout

```text
configs/                         Shared dataset and model settings
data/
  raw/                           Original data location (not committed)
  interim/                       Intermediate files (not committed)
  processed/                     Model-ready data and small fixtures
src/pathsnap/
  data/
    preprocessing/               Teammate's raw-data cleaning, EDA, candidate
                                  generation, road graph, and split pipeline
    build_features.py            Shared model feature preparation
    candidate_gen.py             Alternate candidate-generation helpers
    feature_eng.py               Feature engineering helpers
    legacy_eda.py                 Raw-file EDA helpers used by legacy utilities
  models/                         Candidate scorers and model comparison code
  evaluation/                     Metrics and grouped model comparison
  tuning/                         Bayesian hyperparameter optimization
  decoder/                        HMM and Viterbi route reconstruction
  visualization/                  Folium and other visualization helpers
  service/                        FastAPI backend
  ui/                             Evaluator-facing UI
scripts/                          Preprocessing, training, and evaluation CLIs
artifacts/models/                 Saved model artifacts
artifacts/logs/                   Experiment logs
results/
  eda/                             EDA inventories, plots, and maps
  preprocessing/                  Split manifests and preprocessing samples
  metrics/                         Model evaluation tables and summaries
  figures/                         Comparison charts
  maps/                            Route visualizations
  bayesian/                        Hyperparameter search records and plots
report/                            Final technical report materials
```

Raw data and large generated datasets should stay out of Git. Commit source,
configuration, reproducible experiment records, and only the artifacts needed
to reproduce or run the final system.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Preprocessing and EDA

```bash
python scripts/preprocess.py --data-root "/path/to/Student folder" \
  --out-dir results/preprocessing
python scripts/run_eda.py --data-root "/path/to/Student folder" \
  --out-dir results/eda --map-trip 00000001
```

The candidate table is the shared input to model training. Each row represents
a GPS observation and candidate road segment; `is_ground_truth_arc` is the
training label. Keep folds and evaluation splits grouped by trip. Never use
evaluator-test labels for model or hyperparameter selection.

## Model development

Model implementations use the shared schema in `pathsnap.models.base` and
`configs/models.yaml`: each one is a small file that builds an estimator and
calls `register(ModelSpec(name=..., build=...))`, which plugs it into
`scripts/train_all_models.py` and `pathsnap.evaluation.compare_models` with
no other code changes needed.

- `catboost_model.py` -> `"CatBoost"`
- `lightgbm_model.py` -> `"LightGBM"`
- `extra_trees.py` -> `"ExtraTrees"`
- `random_forest_model.py` -> `"RandomForest"`
- `rbf_svm.py` -> `"RBF_SVM"`

(`extra_tree.py` and the top-level `random_forest.py` are earlier,
standalone explorations with their own feature engineering — not wired into
the shared registry/comparison pipeline.)

```bash
# train + compare all 5 registered models, pick the best by CV log-loss
python scripts/train_all_models.py --data data_processing/outputs/train_candidates.parquet
```

### Model comparison results (real data: 80 train trips, 1.15M candidate rows)

5-fold grouped CV + 80/20 holdout (RBF SVM doesn't scale to the full table,
so the comparison runs on a 6000-GPS-point subsample; the winner is then
refit on the full dataset for the saved production model):

| Model | CV log_loss | CV top1_point_acc | test predict_ms |
|---|---|---|---|
| CatBoost | 0.1585 ± 0.0296 | 0.739 | 1.8 |
| LightGBM | 0.1595 ± 0.0308 | 0.737 | 11.1 |
| ExtraTrees | 0.1609 ± 0.0276 | 0.743 | 128.3 |
| RandomForest | 0.1639 ± 0.0269 | 0.735 | 119.6 |
| RBF_SVM | 0.1981 ± 0.0420 | 0.742 | 996.6 |

All four tree-based models are statistically tied on log-loss (within one CV
std); among those, **CatBoost** is selected for being by far the fastest at
inference (the HMM scores every candidate of every GPS point). RBF SVM is
clearly worse on log-loss and ~500x slower. Saved as
`artifacts/models/catboost_best.joblib`. Held out on the 10 student_test
trips: accuracy 0.944, log_loss 0.136, top1_point_acc 0.782. Full table:
`results/metrics/model_comparison.md`.

Training, comparison, tuning, and decoder experiments should be launched
from scripts in `scripts/`, with measured results saved under `results/`
and model artifacts under `artifacts/models/`.
