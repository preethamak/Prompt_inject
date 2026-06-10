from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


RANDOM_STATE = 42
DEFAULT_DATASET_LIMIT = 10_000
DEFAULT_CONTEXT_MAX_CHARS = 1_500
DEFAULT_LOG_PATH = Path("notebook_results") / "accuracy_runs.jsonl"

_DATASET_CACHE = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "datasets--MAlmasabi--Indirect-Prompt-Injection-BIPIA-GPT"
    / "snapshots"
    / "e5c8011e16ea1d621897a24473b40fd7afccbd92"
    / "dataset_for_huggingface.jsonl"
)

_MODEL_CACHE_HINTS = {
    "BAAI/bge-small-en-v1.5": (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--BAAI--bge-small-en-v1.5"
        / "snapshots"
        / "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    ),
    "sentence-transformers/all-MiniLM-L6-v2": (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--sentence-transformers--all-MiniLM-L6-v2"
    ),
}

_INSTRUCTION_PATTERNS = [
    r"\bignore\b.{0,40}\b(instruction|previous|above|system)\b",
    r"\b(system prompt|developer message)\b",
    r"\b(reveal|bypass|override)\b",
    r"\btool\b.{0,24}\bcall\b",
    r"\bexecute\b.{0,24}\bcommand\b",
]


@dataclass
class SplitBundle:
    x_train: list[str]
    x_val: list[str]
    x_test: list[str]
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    rows_train: list[dict[str, Any]]
    rows_val: list[dict[str, Any]]
    rows_test: list[dict[str, Any]]


def load_bipia_rows(
    limit: int = DEFAULT_DATASET_LIMIT,
    dataset_jsonl: str | Path = _DATASET_CACHE,
) -> list[dict[str, Any]]:
    dataset_path = Path(dataset_jsonl)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Cached dataset not found at {dataset_path}. "
            "Populate the local Hugging Face cache before running notebook experiments."
        )

    rows: list[dict[str, Any]] = []
    with dataset_path.open() as f:
        for idx, line in enumerate(f):
            if idx >= limit:
                break
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows loaded from {dataset_path}")
    return rows


def build_text(row: dict[str, Any], context_max_chars: int = DEFAULT_CONTEXT_MAX_CHARS) -> str:
    return (
        f"Context: {str(row.get('context', ''))[:context_max_chars]}\n"
        f"User intent: {str(row.get('user_intent', ''))}"
    )


def make_split(
    rows: list[dict[str, Any]],
    context_max_chars: int = DEFAULT_CONTEXT_MAX_CHARS,
    random_state: int = RANDOM_STATE,
) -> SplitBundle:
    texts = [build_text(row, context_max_chars=context_max_chars) for row in rows]
    labels = np.array([int(row["label"]) for row in rows], dtype=int)

    x_train, x_temp, y_train, y_temp, rows_train, rows_temp = train_test_split(
        texts,
        labels,
        rows,
        test_size=0.30,
        random_state=random_state,
        stratify=labels,
    )
    x_val, x_test, y_val, y_test, rows_val, rows_test = train_test_split(
        x_temp,
        y_temp,
        rows_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp,
    )
    return SplitBundle(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        rows_train=rows_train,
        rows_val=rows_val,
        rows_test=rows_test,
    )


def build_dense_features(rows: list[dict[str, Any]]) -> np.ndarray:
    patterns = [re.compile(pattern, re.I | re.S) for pattern in _INSTRUCTION_PATTERNS]
    features = []
    for row in rows:
        context = str(row.get("context", ""))
        user_intent = str(row.get("user_intent", ""))
        hits = sum(len(pattern.findall(context)) for pattern in patterns)
        features.append(
            [
                float(np.log1p(len(context))),
                float(np.log1p(len(user_intent))),
                float(np.log1p(context.count("\n"))),
                float(np.log1p(context.count("|"))),
                float(np.log1p(context.count("```"))),
                float(np.log1p(hits)),
                float("?" in user_intent),
            ]
        )
    return np.asarray(features, dtype=float)


def search_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.50
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        y_pred = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_threshold = float(threshold)
            best_f1 = float(f1)
    return best_threshold, best_f1


def summarize_metrics(name: str, y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "model": name,
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "f1": round(float(f1_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred)), 6),
        "recall": round(float(recall_score(y_true, y_pred)), 6),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 6),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 6),
    }


def _build_sparse_text_features(
    split: SplitBundle,
    word_ngram: tuple[int, int],
    char_ngram: tuple[int, int],
    max_word_features: int,
    max_char_features: int,
) -> tuple[csr_matrix, csr_matrix, csr_matrix]:
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=word_ngram,
        min_df=2,
        max_features=max_word_features,
        sublinear_tf=True,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=char_ngram,
        min_df=2,
        max_features=max_char_features,
        sublinear_tf=True,
    )
    x_train_word = word_vec.fit_transform(split.x_train)
    x_val_word = word_vec.transform(split.x_val)
    x_test_word = word_vec.transform(split.x_test)

    x_train_char = char_vec.fit_transform(split.x_train)
    x_val_char = char_vec.transform(split.x_val)
    x_test_char = char_vec.transform(split.x_test)

    return (
        hstack([x_train_word, x_train_char]).tocsr(),
        hstack([x_val_word, x_val_char]).tocsr(),
        hstack([x_test_word, x_test_char]).tocsr(),
    )


def _run_tfidf_experiment(
    split: SplitBundle,
    *,
    name: str,
    word_ngram: tuple[int, int],
    char_ngram: tuple[int, int],
    logreg_c: float,
    class_weight: str | None = None,
    include_dense: bool = False,
    max_word_features: int = 120_000,
    max_char_features: int = 160_000,
) -> dict[str, Any]:
    x_train, x_val, x_test = _build_sparse_text_features(
        split,
        word_ngram=word_ngram,
        char_ngram=char_ngram,
        max_word_features=max_word_features,
        max_char_features=max_char_features,
    )
    if include_dense:
        x_train = hstack([x_train, csr_matrix(build_dense_features(split.rows_train))]).tocsr()
        x_val = hstack([x_val, csr_matrix(build_dense_features(split.rows_val))]).tocsr()
        x_test = hstack([x_test, csr_matrix(build_dense_features(split.rows_test))]).tocsr()

    clf = LogisticRegression(
        C=logreg_c,
        solver="saga",
        max_iter=2500,
        random_state=RANDOM_STATE,
        class_weight=class_weight,
    )
    clf.fit(x_train, split.y_train)

    val_prob = clf.predict_proba(x_val)[:, 1]
    test_prob = clf.predict_proba(x_test)[:, 1]
    threshold, val_f1 = search_best_threshold(split.y_val, val_prob)
    row = summarize_metrics(name, split.y_test, test_prob, threshold)
    row["val_f1"] = round(val_f1, 6)
    row["notes"] = {
        "word_ngram": list(word_ngram),
        "char_ngram": list(char_ngram),
        "logreg_c": logreg_c,
        "class_weight": class_weight,
        "include_dense": include_dense,
    }
    return row


def _resolve_sentence_transformer(model_id: str) -> str:
    hint = _MODEL_CACHE_HINTS.get(model_id)
    if hint and hint.exists():
        if hint.is_dir() and (hint / "config.json").exists():
            return str(hint)
        snapshots = sorted(hint.glob("snapshots/*/config.json"))
        if snapshots:
            return str(snapshots[-1].parent)
    return model_id


def _encode_texts(model_id: str, texts: list[str], batch_size: int = 32) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(_resolve_sentence_transformer(model_id))
    return np.asarray(
        model.encode(texts, batch_size=batch_size, show_progress_bar=True),
        dtype=np.float32,
    )


def _run_embedding_experiment(
    split: SplitBundle,
    *,
    name: str,
    model_id: str,
    xgb_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = {
        "n_estimators": 220,
        "max_depth": 8,
        "learning_rate": 0.08,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
    }
    if xgb_params:
        params.update(xgb_params)

    emb_train = _encode_texts(model_id, split.x_train)
    emb_val = _encode_texts(model_id, split.x_val)
    emb_test = _encode_texts(model_id, split.x_test)

    clf = XGBClassifier(**params)
    clf.fit(emb_train, split.y_train)

    val_prob = clf.predict_proba(emb_val)[:, 1]
    test_prob = clf.predict_proba(emb_test)[:, 1]
    threshold, val_f1 = search_best_threshold(split.y_val, val_prob)
    row = summarize_metrics(name, split.y_test, test_prob, threshold)
    row["val_f1"] = round(val_f1, 6)
    row["notes"] = {
        "model_id": model_id,
        "xgb_params": params,
    }
    return row


def append_experiment_log(rows: list[dict[str, Any]], log_path: str | Path = DEFAULT_LOG_PATH) -> Path:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def load_experiment_log(log_path: str | Path = DEFAULT_LOG_PATH) -> pd.DataFrame:
    path = Path(log_path)
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def run_accuracy_suite(
    *,
    dataset_limit: int = DEFAULT_DATASET_LIMIT,
    context_max_chars: int = DEFAULT_CONTEXT_MAX_CHARS,
    include_embeddings: bool = True,
    fast_mode: bool = False,
    experiment_tag: str = "notebook_accuracy_upgrade_v1",
    log_path: str | Path = DEFAULT_LOG_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = load_bipia_rows(limit=dataset_limit)
    split = make_split(rows, context_max_chars=context_max_chars)

    results: list[dict[str, Any]] = []
    candidates = [
        {
            "name": "tfidf_logreg_baseline",
            "word_ngram": (1, 2),
            "char_ngram": (3, 5),
            "logreg_c": 4.0,
            "include_dense": False,
        },
        {
            "name": "tfidf_logreg_tuned",
            "word_ngram": (1, 3),
            "char_ngram": (3, 6),
            "logreg_c": 8.0,
            "include_dense": False,
        },
        {
            "name": "tfidf_logreg_tuned_plus_dense",
            "word_ngram": (1, 3),
            "char_ngram": (3, 6),
            "logreg_c": 8.0,
            "include_dense": True,
        },
    ]
    if fast_mode:
        candidates = candidates[-2:]

    for candidate in candidates:
        result = _run_tfidf_experiment(split, **candidate)
        result["experiment_tag"] = experiment_tag
        result["dataset_limit"] = dataset_limit
        result["context_max_chars"] = context_max_chars
        result["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        results.append(result)

    if include_embeddings and not fast_mode:
        try:
            result = _run_embedding_experiment(
                split,
                name="bge_small_xgb_tuned",
                model_id="BAAI/bge-small-en-v1.5",
            )
            result["experiment_tag"] = experiment_tag
            result["dataset_limit"] = dataset_limit
            result["context_max_chars"] = context_max_chars
            result["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
            results.append(result)
        except Exception as exc:
            results.append(
                {
                    "model": "bge_small_xgb_tuned",
                    "experiment_tag": experiment_tag,
                    "dataset_limit": dataset_limit,
                    "context_max_chars": context_max_chars,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    append_experiment_log(results, log_path=log_path)
    df = pd.DataFrame(results)
    if "f1" in df.columns:
        df = df.sort_values(["f1", "accuracy"], ascending=False, na_position="last").reset_index(drop=True)

    best_model = None
    ok_rows = [row for row in results if row.get("status") != "error" and "f1" in row]
    if ok_rows:
        best_model = max(ok_rows, key=lambda row: (row["f1"], row["accuracy"]))

    artifact = {
        "log_path": str(Path(log_path)),
        "dataset_rows": len(rows),
        "split_sizes": {
            "train": len(split.x_train),
            "val": len(split.x_val),
            "test": len(split.x_test),
        },
        "best_model": best_model,
    }
    return df, artifact
