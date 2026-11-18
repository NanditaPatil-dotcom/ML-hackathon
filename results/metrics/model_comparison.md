# Candidate-segment model comparison

Data: `/mnt/project-files/data_processing_outputs/train_candidates.parquet` (44898 rows, groups by `trip_id`), positive rate 0.128.

Split: grouped 80/20 holdout by `trip_id`, 5-fold grouped CV on the 80%. Selection metric: CV **log_loss** (lower is better), because the probabilities feed the HMM as emission scores and must be well calibrated.

## Ranking

| # | Model | CV log_loss | CV accuracy | CV top1_point_acc | Test log_loss | Test accuracy | Fit (s) | Inference (ms / 1k candidates) |
|---|---|---|---|---|---|---|---|---|
| 1 | **CatBoost** (selected) | 0.1585 | 0.9337 | 0.7387 | 0.1686 | 0.9286 | 1.85 | 0.25 |
| 2 | LightGBM | 0.1595 | 0.9344 | 0.7373 | 0.1699 | 0.9298 | 0.16 | 1.80 |
| 3 | RandomForest | 0.1721 | 0.9313 | 0.7312 | 0.1857 | 0.9240 | 5.01 | 16.89 |
| 4 | ExtraTrees | 0.1747 | 0.9324 | 0.7316 | 0.1951 | 0.9271 | 1.39 | 17.33 |
| 5 | RBF_SVM | 0.1981 | 0.9331 | 0.7422 | 0.2096 | 0.9292 | 42.21 | 180.80 |

## 5-fold CV (mean ± std)

| Model | accuracy | f1 | roc_auc | log_loss | brier | top1_point_acc |
|---|---|---|---|---|---|---|
| CatBoost | 0.9337 ± 0.0179 | 0.7323 ± 0.0543 | 0.9598 ± 0.0097 | 0.1585 ± 0.0296 | 0.0470 ± 0.0110 | 0.7387 ± 0.0765 |
| LightGBM | 0.9344 ± 0.0186 | 0.7352 ± 0.0504 | 0.9592 ± 0.0099 | 0.1595 ± 0.0308 | 0.0471 ± 0.0113 | 0.7373 ± 0.0755 |
| RandomForest | 0.9313 ± 0.0174 | 0.7377 ± 0.0475 | 0.9572 ± 0.0094 | 0.1721 ± 0.0281 | 0.0495 ± 0.0107 | 0.7312 ± 0.0760 |
| ExtraTrees | 0.9324 ± 0.0172 | 0.7174 ± 0.0511 | 0.9541 ± 0.0112 | 0.1747 ± 0.0320 | 0.0496 ± 0.0112 | 0.7316 ± 0.0769 |
| RBF_SVM | 0.9331 ± 0.0200 | 0.6994 ± 0.0666 | 0.9260 ± 0.0123 | 0.1981 ± 0.0420 | 0.0528 ± 0.0146 | 0.7422 ± 0.0803 |

## Holdout test set

| Model | accuracy | f1 | roc_auc | log_loss | brier | top1_point_acc | fit_s | predict_ms |
|---|---|---|---|---|---|---|---|---|
| CatBoost | 0.9286 | 0.7370 | 0.9591 | 0.1686 | 0.0504 | 0.7607 | 1.85 | 2.1 |
| LightGBM | 0.9298 | 0.7375 | 0.9585 | 0.1699 | 0.0504 | 0.7616 | 0.16 | 15.0 |
| RandomForest | 0.9240 | 0.7334 | 0.9539 | 0.1857 | 0.0535 | 0.7539 | 5.01 | 140.9 |
| ExtraTrees | 0.9271 | 0.7231 | 0.9503 | 0.1951 | 0.0528 | 0.7564 | 1.39 | 144.5 |
| RBF_SVM | 0.9292 | 0.7113 | 0.9265 | 0.2096 | 0.0564 | 0.7659 | 42.21 | 1508.3 |

## Selected model: **CatBoost**

Rule: models within one CV std (0.0296) of the best CV log_loss (CatBoost, 0.1585) count as tied: CatBoost, LightGBM, RandomForest, ExtraTrees. Among those the fastest at inference wins: **CatBoost** (CV log_loss 0.1585, 2.1 ms on the test set).

Metric notes: `top1_point_acc` = share of GPS points where the highest-probability candidate is the true road (what the HMM effectively relies on). Accuracy/F1 use a 0.5 threshold.
