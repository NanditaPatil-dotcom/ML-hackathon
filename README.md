# ML-hackathon

## Candidate-segment model training (CatBoost + RBF SVM)

Trains and compares model families that score each GPS point's candidate road
segments, for the HMM/Viterbi step downstream. This script covers **CatBoost**
and **RBF SVM**; teammates are covering LightGBM, Extra Trees and Random Forest
in parallel against the same interface, so the full 5-model comparison the
task spec requires comes together from both.

Real preprocessing isn't ready yet, so this is built and tested against a
mocked dataset with the same column names the real pipeline will produce:
`distance_to_segment, bearing_diff, speed_limit, transition_distance,
candidate_road_id, label` (plus `point_id`/`trip_id` for grouped splits).

```bash
pip install -r requirements.txt

# generate the mock dataset
python src/generate_mock_data.py

# train + compare, write outputs/model_comparison.{csv,md} and outputs/best_model.joblib
python src/train_compare.py --data data/mock/candidates_mock.csv --out-dir outputs

# once real preprocessed data lands, just point at it (same column names):
python src/train_compare.py --data path/to/real_preprocessed.csv --out-dir outputs
```

### Results on the mock dataset

| Model | CV log_loss | CV top1_point_acc | test predict_ms |
|---|---|---|---|
| RBF_SVM | 0.2223 ± 0.0125 | 0.858 | 60.9 |
| CatBoost | 0.2223 ± 0.0149 | 0.859 | 1.5 |

Statistically tied on log-loss; **CatBoost** wins on inference speed (needed
since the HMM scores every candidate of every GPS point) and is saved as
`outputs/best_model.joblib`.

### Using the trained model (for the HMM step)

```python
from candidate_scorer import CandidateScorer

scorer = CandidateScorer.load("outputs/best_model.joblib")
df["p_match"] = scorer.predict_proba(df)        # P(candidate is the true road), in [0,1]
df["emission"] = scorer.emission_log_prob(df)   # log-prob, normalized per point_id
```
