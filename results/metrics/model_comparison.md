# Candidate-segment model comparison

Data: `/mnt/project-files/data_processing_outputs/train_candidates.parquet` (44898 rows, groups by `trip_id`), positive rate 0.128.

Split: grouped 80/20 holdout by `trip_id`, 5-fold grouped CV on the 80%. Selection metric: CV **log_loss** (lower is better), because the probabilities feed the HMM as emission scores and must be well calibrated.

## 5-fold CV (mean ± std)

| Model | accuracy | f1 | roc_auc | log_loss | brier | top1_point_acc |
|---|---|---|---|---|---|---|
| CatBoost | 0.9337 ± 0.0179 | 0.7323 ± 0.0543 | 0.9598 ± 0.0097 | 0.1585 ± 0.0296 | 0.0470 ± 0.0110 | 0.7387 ± 0.0765 |
| LightGBM | 0.9344 ± 0.0186 | 0.7352 ± 0.0504 | 0.9592 ± 0.0099 | 0.1595 ± 0.0308 | 0.0471 ± 0.0113 | 0.7373 ± 0.0755 |
| ExtraTrees | 0.9357 ± 0.0174 | 0.7257 ± 0.0538 | 0.9595 ± 0.0097 | 0.1609 ± 0.0276 | 0.0473 ± 0.0107 | 0.7432 ± 0.0798 |
| RandomForest | 0.9342 ± 0.0171 | 0.7282 ± 0.0535 | 0.9581 ± 0.0088 | 0.1639 ± 0.0269 | 0.0475 ± 0.0106 | 0.7348 ± 0.0753 |
| RBF_SVM | 0.9331 ± 0.0200 | 0.6994 ± 0.0666 | 0.9260 ± 0.0123 | 0.1981 ± 0.0420 | 0.0528 ± 0.0146 | 0.7422 ± 0.0803 |

## Holdout test set

| Model | accuracy | f1 | roc_auc | log_loss | brier | top1_point_acc | fit_s | predict_ms |
|---|---|---|---|---|---|---|---|---|
| CatBoost | 0.9286 | 0.7370 | 0.9591 | 0.1686 | 0.0504 | 0.7607 | 1.67 | 1.8 |
| LightGBM | 0.9298 | 0.7375 | 0.9585 | 0.1699 | 0.0504 | 0.7616 | 0.15 | 11.1 |
| ExtraTrees | 0.9307 | 0.7327 | 0.9563 | 0.1718 | 0.0507 | 0.7710 | 1.22 | 128.3 |
| RandomForest | 0.9302 | 0.7383 | 0.9538 | 0.1776 | 0.0512 | 0.7573 | 3.36 | 119.6 |
| RBF_SVM | 0.9292 | 0.7113 | 0.9265 | 0.2096 | 0.0564 | 0.7659 | 36.11 | 996.6 |

## Selected model: **CatBoost**

Rule: models within one CV std (0.0296) of the best CV log_loss (CatBoost, 0.1585) count as tied: CatBoost, LightGBM, ExtraTrees, RandomForest. Among those the fastest at inference wins: **CatBoost** (CV log_loss 0.1585, 1.8 ms on the test set).

Metric notes: `top1_point_acc` = share of GPS points where the highest-probability candidate is the true road (what the HMM effectively relies on). Accuracy/F1 use a 0.5 threshold.
