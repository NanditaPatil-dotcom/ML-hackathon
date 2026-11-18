"""Server-side evaluation against the reference .route (never sent to the UI)."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence


def _lcs(a: Sequence[int], b: Sequence[int]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def route_metrics(pred_route: List[int], per_obs: List[Optional[int]], ref: List[int]) -> Dict[str, float]:
    ref_set, pred_set = set(ref), set(pred_route)
    tp = len(ref_set & pred_set)
    prec = tp / len(pred_set) if pred_set else 0.0
    rec = tp / len(ref_set) if ref_set else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    matched = [a for a in per_obs if a is not None]
    # Observation-level: share of matched GPS points whose arc lies on the reference route.
    obs_acc = sum(a in ref_set for a in matched) / len(matched) if matched else 0.0
    lcs = _lcs(pred_route, ref)
    return {
        "observation_accuracy": round(obs_acc, 4),
        "arc_precision": round(prec, 4),
        "arc_recall": round(rec, 4),
        "arc_f1": round(f1, 4),
        "sequence_similarity": round(2 * lcs / (len(pred_route) + len(ref)), 4) if pred_route or ref else 0.0,
    }
