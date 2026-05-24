"""
Error mining: extract hard false negatives and false positives from test set.
Exports a JSON file per run for targeted data augmentation review.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from .leaderboard import RESULTS_DIR


def get_hard_false_negatives(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    rows: list,
    threshold: float,
    top_n: int = 30,
) -> list[dict]:
    """
    Hard FN: actual=1, predicted=0.
    Sorted by ascending y_prob (model was most confident they were safe).
    """
    fn_mask = (y_true == 1) & (y_prob < threshold)
    indices = np.where(fn_mask)[0]
    indices = indices[np.argsort(y_prob[indices])][:top_n]

    return [
        {"idx": int(i), "true_label": 1, "y_prob": round(float(y_prob[i]), 4),
         "text": rows[i].get("context", "")[:300],
         "intent": rows[i].get("user_intent", "")}
        for i in indices
    ]


def get_hard_false_positives(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    rows: list,
    threshold: float,
    top_n: int = 30,
) -> list[dict]:
    """
    Hard FP: actual=0, predicted=1.
    Sorted by descending y_prob (model was most confident they were injections).
    """
    fp_mask = (y_true == 0) & (y_prob >= threshold)
    indices = np.where(fp_mask)[0]
    indices = indices[np.argsort(-y_prob[indices])][:top_n]

    return [
        {"idx": int(i), "true_label": 0, "y_prob": round(float(y_prob[i]), 4),
         "text": rows[i].get("context", "")[:300],
         "intent": rows[i].get("user_intent", "")}
        for i in indices
    ]


def export_hard_examples(
    run_id: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    rows: list,
    threshold: float,
    top_n: int = 30,
) -> Path:
    """
    Save hard FNs and FPs to autoresearch_results/errors_{run_id}.json.
    Returns the path of the saved file.
    """
    out = {
        "run_id": run_id,
        "threshold": threshold,
        "hard_false_negatives": get_hard_false_negatives(y_true, y_prob, rows, threshold, top_n),
        "hard_false_positives": get_hard_false_positives(y_true, y_prob, rows, threshold, top_n),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"errors_{run_id}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"Saved hard examples → {path}")
    return path
