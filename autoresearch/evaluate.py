"""
Evaluation utilities.

Objective hierarchy (locked per plan):
  Primary  : val_f1
  Secondary: tpr @ 0.1% FPR
  Test     : computed only for top-kept runs
"""

from __future__ import annotations
import re
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
) -> tuple[float, float]:
    """
    Find the exact threshold on y_prob that maximises F1 subject to FPR <= max_fpr.
    Returns (best_threshold, best_val_f1).
    """
    scores = np.asarray(y_prob, dtype=float)
    unique_scores = np.unique(scores)
    if unique_scores.size == 0:
        return 0.5, 0.0

    candidates = np.concatenate((
        [np.nextafter(unique_scores.max(), np.inf)],
        unique_scores[::-1],
    ))

    best_thr, best_f1, best_fpr = 0.5, -1.0, float("inf")
    for thr in candidates:
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn + 1e-12)
        if fpr > max_fpr:
            continue
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if (
            f1 > best_f1
            or (np.isclose(f1, best_f1) and fpr < best_fpr)
            or (np.isclose(f1, best_f1) and np.isclose(fpr, best_fpr) and thr > best_thr)
        ):
            best_f1, best_thr, best_fpr = f1, float(thr), fpr

    if best_f1 < 0:
        return float(np.nextafter(unique_scores.max(), np.inf)), 0.0
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


def _safe_slice_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    prefix: str,
) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        f"{prefix}_count": int(len(y_true)),
        f"{prefix}_positive_rate": round(float(np.mean(y_true)), 6) if len(y_true) else 0.0,
        f"{prefix}_pred_positive_rate": round(float(np.mean(y_pred)), 6) if len(y_pred) else 0.0,
    }
    if len(np.unique(y_true)) < 2:
        out[f"{prefix}_accuracy"] = round(float(accuracy_score(y_true, y_pred)), 6)
        out[f"{prefix}_f1"] = round(float(f1_score(y_true, y_pred, zero_division=0)), 6)
        return out
    out.update(compute_metrics(y_true, y_pred, y_prob, prefix=prefix))
    return out


def compute_slice_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    rows: list[dict],
    threshold: float,
    prefix: str = "",
) -> dict:
    """
    Compute metrics for a few deployment-relevant slices.

    Slices are heuristic, but stable enough to surface regressions on:
    - short/long context
    - table-heavy vs narrative context
    - question-like vs instruction-like intents
    """
    p = f"{prefix}_" if prefix else ""
    contexts = [str(r.get("context", "")) for r in rows]
    intents = [str(r.get("user_intent", "")) for r in rows]
    context_len = np.array([len(c) for c in contexts])

    question_re = re.compile(r"^(what|which|who|when|where|why|how|did|does|is|are|can)\b", re.I)
    instruction_re = re.compile(
        r"^(summarize|list|extract|write|tell|give|show|describe|explain|ignore|return|provide)\b",
        re.I,
    )

    masks = {
        "short_context": context_len <= 300,
        "long_context": context_len >= 1000,
        "table_context": np.array(["|" in c for c in contexts]),
        "narrative_context": np.array(["|" not in c for c in contexts]),
        "question_intent": np.array([
            ("?" in intent) or bool(question_re.match(intent.strip()))
            for intent in intents
        ]),
        "instruction_intent": np.array([
            bool(instruction_re.match(intent.strip()))
            for intent in intents
        ]),
    }

    out: dict[str, float | int] = {}
    for name, mask in masks.items():
        if not mask.any():
            continue
        out.update(_safe_slice_metrics(
            y_true[mask],
            y_prob[mask],
            threshold,
            prefix=f"{p}{name}",
        ))
    return out
