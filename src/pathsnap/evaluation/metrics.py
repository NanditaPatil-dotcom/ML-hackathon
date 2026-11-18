"""Metrics for the candidate-segment scoring task.

Standard classification metrics plus `top1_point_acc`, which is the one that
actually tracks what the HMM cares about: for each GPS point (grouped by
trip_id + point_idx together, since point_idx resets per trip), does the
highest-probability candidate equal the true road?
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                             log_loss, precision_score, recall_score,
                             roc_auc_score)

from pathsnap.models.base import GROUP_COL, LABEL, POINT_COL


def top1_point_accuracy(df: pd.DataFrame, proba: np.ndarray) -> float:
    if GROUP_COL not in df.columns or POINT_COL not in df.columns:
        return float("nan")
    tmp = df[[GROUP_COL, POINT_COL, LABEL]].copy()
    tmp["p"] = proba
    best = tmp.loc[tmp.groupby([GROUP_COL, POINT_COL])["p"].idxmax()]
    return float(best[LABEL].mean())


def compute_metrics(y: np.ndarray, proba: np.ndarray, df: pd.DataFrame, thr: float = 0.5) -> dict:
    pred = (proba >= thr).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
        "log_loss": log_loss(y, np.clip(proba, 1e-7, 1 - 1e-7)),
        "brier": brier_score_loss(y, proba),
        "top1_point_acc": top1_point_accuracy(df, proba),
    }
