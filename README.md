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
`configs/models.yaml`. Extra Trees is implemented in
`src/pathsnap/models/extra_tree.py`. Training, comparison, tuning, and decoder
experiments should be launched from scripts in `scripts/`, with measured
results saved under `results/` and model artifacts under `artifacts/models/`.
