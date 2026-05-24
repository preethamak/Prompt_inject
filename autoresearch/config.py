"""
Evaluation objectives and search space for AutoResearch sweep.

Objective hierarchy (locked):
  1. Primary   : val_f1   (higher is better)
  2. Secondary : tpr@0.1fpr (higher is better under FPR cap)
  3. Test metrics: computed ONLY for runs that improve val_f1 over baseline
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

# ── Reproducibility ─────────────────────────────────────────────────────────
RANDOM_STATE = 42
DATASET_SIZE = 10000
CONTEXT_MAX_CHARS = 1500
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15           # remaining 0.15 goes to test
DATASET_ID = "MAlmasabi/Indirect-Prompt-Injection-BIPIA-GPT"

# ── Objective settings ───────────────────────────────────────────────────────
MAX_FPR_FOR_THRESHOLD = 0.10   # max allowed FPR when selecting threshold on val
FPR_OPERATING_POINTS = (0.001, 0.005, 0.01, 0.05, 0.10)
PRIMARY_METRIC = "val_f1"
BASELINE_F1 = 0.90            # val_f1 threshold; notebook showed ~0.91 test F1

# ── Candidate search spaces (used by runner.generate_candidates) ─────────────
TFIDF_WORD_NGRAMS   = [(1, 1), (1, 2), (1, 3)]
TFIDF_CHAR_NGRAMS   = [(3, 5), (4, 6), (3, 6)]
LOGREG_C_VALUES     = [1.0, 2.0, 4.0, 8.0]

EMBED_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
]

XGB_CONFIGS = [
    {"n_estimators": 350, "max_depth": 7, "learning_rate": 0.05,
     "subsample": 0.90, "colsample_bytree": 0.90, "reg_lambda": 1.0},
    {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.08,
     "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 2.0},
    {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.05,
     "subsample": 0.80, "colsample_bytree": 0.80, "reg_lambda": 1.0},
]

FUSION_META_C = [0.5, 1.0, 2.0]


@dataclass
class RunConfig:
    """Single experiment candidate configuration."""
    run_id: str = ""
    word_ngram: tuple = (1, 2)
    char_ngram: tuple = (3, 5)
    tfidf_max_word_features: int = 120000
    tfidf_max_char_features: int = 160000
    logreg_c: float = 4.0
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    xgb: dict = field(default_factory=lambda: XGB_CONFIGS[0])
    fusion_c: float = 1.0
    context_max_chars: int = CONTEXT_MAX_CHARS
    max_fpr: float = MAX_FPR_FOR_THRESHOLD

    def as_dict(self) -> dict[str, Any]:
        d = {
            "run_id": self.run_id,
            "word_ngram": list(self.word_ngram),
            "char_ngram": list(self.char_ngram),
            "tfidf_max_word_features": self.tfidf_max_word_features,
            "tfidf_max_char_features": self.tfidf_max_char_features,
            "logreg_c": self.logreg_c,
            "embed_model": self.embed_model,
            "fusion_c": self.fusion_c,
            "context_max_chars": self.context_max_chars,
            "max_fpr": self.max_fpr,
        }
        d.update({f"xgb_{k}": v for k, v in self.xgb.items()})
        return d
