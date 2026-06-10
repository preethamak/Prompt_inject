"""
Low-compute routing between TF-IDF and fusion predictions.

Idea:
  - trust TF-IDF on confident samples
  - use fusion only near the TF-IDF uncertainty band
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .evaluate import select_threshold


@dataclass
class RoutingSelection:
    margin: float
    tfidf_threshold: float
    final_threshold: float
    val_f1: float
    used_fusion_fraction: float


def build_routed_probabilities(
    tfidf_prob: np.ndarray,
    fusion_prob: np.ndarray,
    tfidf_threshold: float,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Use TF-IDF except when its probability is within `margin` of its operating threshold.
    Returns (routed_probabilities, use_fusion_mask).
    """
    tfidf_prob = np.asarray(tfidf_prob, dtype=float)
    fusion_prob = np.asarray(fusion_prob, dtype=float)
    use_fusion = np.abs(tfidf_prob - tfidf_threshold) <= margin
    routed = np.where(use_fusion, fusion_prob, tfidf_prob)
    return routed, use_fusion


def select_routing_strategy(
    y_true: np.ndarray,
    tfidf_prob: np.ndarray,
    fusion_prob: np.ndarray,
    tfidf_threshold: float,
    max_fpr: float,
    margin_grid: tuple[float, ...],
) -> RoutingSelection:
    """
    Pick the routing margin that gives the best validation F1 under the global threshold rule.
    """
    best = RoutingSelection(
        margin=0.0,
        tfidf_threshold=tfidf_threshold,
        final_threshold=tfidf_threshold,
        val_f1=-1.0,
        used_fusion_fraction=0.0,
    )

    for margin in margin_grid:
        routed_prob, use_fusion = build_routed_probabilities(
            tfidf_prob=tfidf_prob,
            fusion_prob=fusion_prob,
            tfidf_threshold=tfidf_threshold,
            margin=margin,
        )
        threshold, f1 = select_threshold(y_true, routed_prob, max_fpr=max_fpr)
        use_fusion_fraction = float(np.mean(use_fusion)) if len(use_fusion) else 0.0
        if (
            f1 > best.val_f1
            or (np.isclose(f1, best.val_f1) and use_fusion_fraction < best.used_fusion_fraction)
        ):
            best = RoutingSelection(
                margin=float(margin),
                tfidf_threshold=float(tfidf_threshold),
                final_threshold=float(threshold),
                val_f1=float(f1),
                used_fusion_fraction=use_fusion_fraction,
            )

    return best
