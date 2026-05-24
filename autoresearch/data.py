"""
Deterministic data loading and train/val/test split.
Split is always fixed by RANDOM_STATE to ensure comparability across runs.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from datasets import load_dataset

from .config import (
    RANDOM_STATE, DATASET_SIZE, DATASET_ID,
    TRAIN_FRAC, VAL_FRAC, CONTEXT_MAX_CHARS,
)


@dataclass
class DataSplit:
    """Holds all three splits as plain lists/arrays."""
    x_train: list
    x_val: list
    x_test: list
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    rows_test: list          # raw dataset rows for test set (for error mining)

    @property
    def sizes(self) -> str:
        return (f"train={len(self.x_train)}, "
                f"val={len(self.x_val)}, "
                f"test={len(self.x_test)}")


def _build_text(row: dict, context_max_chars: int) -> str:
    return f"Context: {row['context'][:context_max_chars]}\nUser intent: {row['user_intent']}"


def load_split(
    dataset_size: int = DATASET_SIZE,
    context_max_chars: int = CONTEXT_MAX_CHARS,
    random_state: int = RANDOM_STATE,
) -> DataSplit:
    """Load dataset and return a deterministic 70/15/15 stratified split."""
    ds = load_dataset(DATASET_ID, split=f"train[:{dataset_size}]")
    texts = [_build_text(r, context_max_chars) for r in ds]
    y = np.array([int(r["label"]) for r in ds])
    rows = list(ds)

    test_size = 1.0 - TRAIN_FRAC              # 0.30

    x_train, x_temp, y_train, y_temp, rows_train, rows_temp = train_test_split(
        texts, y, rows,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    val_ratio = VAL_FRAC / test_size          # 0.15 / 0.30 = 0.50
    x_val, x_test, y_val, y_test, _, rows_test = train_test_split(
        x_temp, y_temp, rows_temp,
        test_size=(1.0 - val_ratio),
        random_state=random_state,
        stratify=y_temp,
    )

    return DataSplit(
        x_train=x_train, x_val=x_val, x_test=x_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        rows_test=rows_test,
    )
