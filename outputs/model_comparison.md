# Candidate-segment model comparison

Data: `../data/mock/candidates_mock.csv` (9043 rows, 40 groups by `trip_id`), positive rate 0.221.

Split: grouped 80/20 holdout by `trip_id`, 5-fold grouped CV on the 80%. Selection metric: CV **log_loss** (lower is better), because the probabilities feed the HMM as emission scores and must be well calibrated.

## 5-fold CV (mean ± std)

| Model | accuracy | f1 | roc_auc | log_loss | brier | top1_point_acc |
|---|---|---|---|---|---|---|
| RBF_SVM | 0.9018 ± 0.0079 | 0.7794 ± 0.0193 | 0.9550 ± 0.0061 | 0.2223 ± 0.0125 | 0.0698 ± 0.0040 | 0.8580 ± 0.0247 |
| CatBoost | 0.8995 ± 0.0107 | 0.7759 ± 0.0249 | 0.9552 ± 0.0061 | 0.2223 ± 0.0149 | 0.0706 ± 0.0050 | 0.8592 ± 0.0130 |

## Holdout test set

| Model | accuracy | f1 | roc_auc | log_loss | brier | top1_point_acc | fit_s | predict_ms |
|---|---|---|---|---|---|---|---|---|
| RBF_SVM | 0.9012 | 0.7879 | 0.9531 | 0.2267 | 0.0704 | 0.8500 | 1.35 | 60.9 |
| CatBoost | 0.9029 | 0.7938 | 0.9536 | 0.2266 | 0.0707 | 0.8450 | 1.36 | 1.5 |

## Selected model: **CatBoost**

Rule: models within one CV std (0.0125) of the best CV log_loss (RBF_SVM, 0.2223) count as tied: RBF_SVM, CatBoost. Among those the fastest at inference wins: **CatBoost** (CV log_loss 0.2223, 1.5 ms on the test set). Saved to `best_model.joblib`.

Metric notes: `top1_point_acc` = share of GPS points where the highest-probability candidate is the true road (what the HMM effectively relies on). Accuracy/F1 use a 0.5 threshold.
