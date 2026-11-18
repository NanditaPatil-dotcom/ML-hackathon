# PathSnap

PathSnap maps noisy GPS observations to candidate road segments, scores those
candidates with machine-learning models, and reconstructs a connected route
with an HMM/Viterbi decoder.

## Repository layout

```text
configs/                         Dataset and model configuration
data/
  raw/                           Place raw input data here (not committed)
  interim/                       Intermediate data (not committed)
  processed/                     Small fixtures and mock data
src/pathsnap/
  data/preprocessing/            Current raw-data cleaning, features, EDA,
                                 road graph, route matching, and pipeline
  data/                           Shared data helpers and inference candidate
                                 generation
  models/                         Canonical model registry and five models
  evaluation/                     Metrics and grouped comparison
  tuning/                         Bayesian hyperparameter optimization
  decoder/                        HMM/Viterbi route reconstruction
  service/                        FastAPI backend
  ui/                             Streamlit evaluator interface
  visualization/                  Folium visualization helpers
  legacy/                          Retained standalone/earlier experiments
scripts/
  preprocess.py                   Build train and student-test candidate data
  run_eda.py                      Generate dataset inventory, plots, and map
  train_all_models.py             Train/compare the registered model suite
  evaluate_student_test.py        Student-test metrics and comparison plots
  evaluate_student_test_diagnostics.py
                                  Confusion matrices and sequence metrics
  run_server.sh / run_ui.sh       Start the API and UI
  experiments/                    Earlier one-model/report utilities
artifacts/
  models/                          Current model artifacts
  logs/                            Training logs
  archive/                         Older model artifacts retained for reference
results/
  preprocessing/                  Generated candidate datasets and manifests
  eda/                             EDA tables, plots, and maps
  metrics/                         Evaluation tables and summaries
  figures/                         Model and decoder comparison plots
  maps/                            Interactive route maps
  bayesian/                        Current optimization outputs
  archive/                         Earlier metrics, maps, and experiment outputs
report/                            Consolidated technical report files
PathSnap_student_materials/        Assignment reference PDFs
tests/                             Focused decoder tests
```

Raw datasets, large generated candidate tables, and model binaries may be
gitignored. Keep source, configuration, reproducible result summaries, and the
inference artifact needed for the final demo in the repository.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Commands below assume the repository root as the working directory.

## Preprocessing and EDA

```bash
python scripts/preprocess.py \
  --data-root "/path/to/Student folder" \
  --out-dir results/preprocessing/candidate_data

python scripts/run_eda.py \
  --data-root "/path/to/Student folder" \
  --out-dir results/eda \
  --map-trip 00000001
```

The preprocessing output is the shared input to the current model suite. Rows
represent observation/candidate-road pairs; the label is
`is_ground_truth_arc`. Keep all data splits grouped by `trip_id`. The reserved
evaluator-test partition must not be used for development or model selection.

## Model comparison and evaluation

The active registry contains CatBoost, LightGBM, ExtraTrees, Random Forest,
and RBF SVM. The model implementation in `src/pathsnap/models/extra_trees.py`
is the sole ExtraTrees registry entry; old wrappers and standalone experiments
are retained under `src/pathsnap/legacy/`.

```bash
python scripts/train_all_models.py \
  --data results/preprocessing/candidate_data/train_candidates.parquet

python scripts/evaluate_student_test.py \
  --data-root "/path/to/Student folder"

python scripts/evaluate_student_test_diagnostics.py \
  --data-root "/path/to/Student folder"
```

Training-side CV and model selection use training trips. The ten
`student_test` trips support development-stage comparison only. The separate
evaluator-test partition must remain untouched. Detailed outputs are saved in
`results/metrics/` and `results/figures/`.

## API and UI

The launch scripts are:

```bash
bash scripts/run_server.sh
bash scripts/run_ui.sh
```

Before the evaluator demo, configure the backend's dataset and model locations
for the deployment machine and verify the API and UI against the final model
artifact. The API implementation currently includes machine-specific paths;
do not assume the launchers alone make the service portable. The older root
artifact `artifacts/best_model.joblib` is retained for compatibility; the
selected comparison artifact is under `artifacts/models/`.

## Reports and experiment history

The five assignment PDFs are kept in `PathSnap_student_materials/`. The
technical report is in `report/`. Rebuild the report only from current saved
metrics and verified experiment outputs. The script in
`scripts/experiments/generate_report_legacy.py` is an archived earlier version
and contains machine-specific paths; it is not the current report generator.

Earlier model and benchmark implementations are kept in `src/pathsnap/legacy/`
for provenance. They are not imported by the active comparison pipeline.
Older generated comparisons and optimization outputs are in
`results/archive/`; use the `student_test_*` files in `results/metrics/` for the
latest student-test evaluation.
