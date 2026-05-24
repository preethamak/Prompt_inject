"""
Evaluation utilities.

Objective hierarchy (locked per plan):
  Primary  : val_f1
  Secondary: tpr @ 0.1% FPR
  Test     : computed only for top-kept runs
"""

from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix, roc_curve,
)
from .config import FPR_OPERATING_POINTS, MAX_FPR_FOR_THRESHOLD


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    prefix: str = "",
) -> dict:
    """Return full metric dict: accuracy, f1, precision, recall, roc_auc, pr_auc."""
    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        f"{p}f1":       round(float(f1_score(y_true, y_pred)), 6),
        f"{p}precision":round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        f"{p}recall":   round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        f"{p}roc_auc":  round(float(roc_auc_score(y_true, y_prob)), 6),
        f"{p}pr_auc":   round(float(average_precision_score(y_true, y_prob)), 6),
    }


def compute_tpr_at_fpr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fpr_targets: tuple = FPR_OPERATING_POINTS,
    prefix: str = "",
) -> dict:
    """Return TPR at each requested FPR operating point."""
    fprs, tprs, _ = roc_curve(y_true, y_prob)
    p = f"{prefix}_" if prefix else ""
    out = {}
    for fpr_target in fpr_targets:
        mask = fprs <= fpr_target
        tpr_val = float(tprs[mask].max()) if mask.any() else 0.0
        key = f"{p}tpr_at_fpr_{fpr_target:.4f}"
        out[key] = round(tpr_val, 6)
    return out


def select_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    max_fpr: float = MAX_FPR_FOR_THRESHOLD,
    n_steps: int = 199,
) -> tuple[float, float]:
    """
    Find threshold on y_prob that maximises F1 subject to FPR <= max_fpr.
    Returns (best_threshold, best_val_f1).
    """
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.01, 0.99, n_steps):
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn + 1e-12)
        if fpr > max_fpr:
            continue
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr, best_f1


def evaluate_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    prefix: str = "",
) -> dict:
    """Compute full metrics + TPR@FPR at the given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    m = compute_metrics(y_true, y_pred, y_prob, prefix=prefix)
    m.update(compute_tpr_at_fpr(y_true, y_prob, prefix=prefix))
    p = f"{prefix}_" if prefix else ""
    m[f"{p}threshold"] = round(threshold, 4)
    return m
