"""
Feature builders: TF-IDF (word + char) and sentence embeddings.

Embedding caching strategy (two layers):
  1. Disk cache (autoresearch_results/embed_cache/): persisted across Python sessions.
     Key = md5(model_id + n_texts + first/last text hash). Saved as .npy files.
  2. In-memory cache: avoids repeated disk reads within a single sweep.
"""

from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from .config import RANDOM_STATE

_CACHE_DIR = Path(__file__).parent.parent / "autoresearch_results" / "embed_cache"
_MEMORY_CACHE: dict[str, np.ndarray] = {}


# ── TF-IDF ───────────────────────────────────────────────────────────────────

class TfidfBranch:
    """Word + char TF-IDF stacked into one sparse matrix, plus LogReg classifier."""

    def __init__(
        self,
        word_ngram=(1, 2),
        char_ngram=(3, 5),
        max_word_features=120_000,
        max_char_features=160_000,
        logreg_c=4.0,
        random_state=RANDOM_STATE,
    ):
        self.word_vec = TfidfVectorizer(
            analyzer="word", ngram_range=word_ngram,
            min_df=2, max_features=max_word_features, sublinear_tf=True,
        )
        self.char_vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=char_ngram,
            min_df=2, max_features=max_char_features, sublinear_tf=True,
        )
        self.clf = LogisticRegression(
            C=logreg_c, solver="saga",
            max_iter=4000, random_state=random_state,
        )

    def fit_transform(self, x_train, y_train):
        x_word = self.word_vec.fit_transform(x_train)
        x_char = self.char_vec.fit_transform(x_train)
        x = hstack([x_word, x_char]).tocsr()
        self.clf.fit(x, y_train)
        return x

    def transform(self, texts):
        return hstack([
            self.word_vec.transform(texts),
            self.char_vec.transform(texts),
        ]).tocsr()

    def predict_proba(self, texts) -> np.ndarray:
        return self.clf.predict_proba(self.transform(texts))[:, 1]


# ── Embedding + XGBoost ──────────────────────────────────────────────────────

def _cache_key(model_id: str, texts: list) -> str:
    snippet = "".join(texts[:3] + texts[-3:])
    h = hashlib.md5(f"{model_id}|{len(texts)}|{snippet}".encode()).hexdigest()[:16]
    return f"{model_id.replace('/', '_')}_{len(texts)}_{h}"


def _get_embeddings(model_id: str, texts: list, batch_size: int = 64) -> np.ndarray:
    """Load from memory cache → disk cache → encode fresh."""
    from sentence_transformers import SentenceTransformer  # lazy import
    key = _cache_key(model_id, texts)

    if key in _MEMORY_CACHE:
        return _MEMORY_CACHE[key]

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    disk_path = _CACHE_DIR / f"{key}.npy"
    if disk_path.exists():
        print(f"  [embed cache hit] {disk_path.name}")
        arr = np.load(str(disk_path))
        _MEMORY_CACHE[key] = arr
        return arr

    print(f"  [encoding] {model_id}  n={len(texts)}  batch={batch_size}")
    model = SentenceTransformer(model_id)
    arr = model.encode(texts, show_progress_bar=True, batch_size=batch_size)
    np.save(str(disk_path), arr)
    _MEMORY_CACHE[key] = arr
    print(f"  [saved] {disk_path.name}")
    return arr


def clear_embed_cache(disk: bool = False):
    """Clear in-memory cache. Optionally delete disk cache too."""
    _MEMORY_CACHE.clear()
    if disk:
        import shutil
        if _CACHE_DIR.exists():
            shutil.rmtree(_CACHE_DIR)
            print(f"Cleared disk embed cache: {_CACHE_DIR}")


class EmbedXGBBranch:
    """Sentence embeddings + XGBoost classifier."""

    def __init__(self, model_id: str, xgb_params: dict, random_state=RANDOM_STATE):
        self.model_id = model_id
        params = dict(xgb_params)
        params.setdefault("eval_metric", "logloss")
        params["random_state"] = random_state
        self.clf = XGBClassifier(**params)

    def fit(self, x_train: list, y_train, x_val: list = None):
        emb_train = _get_embeddings(self.model_id, x_train)
        self.clf.fit(emb_train, y_train)

    def predict_proba(self, texts: list) -> np.ndarray:
        emb = _get_embeddings(self.model_id, texts)
        return self.clf.predict_proba(emb)[:, 1]


# ── Fusion meta-model ────────────────────────────────────────────────────────

class FusionModel:
    """Stacks branch probabilities and trains a LogReg meta-model on val set."""

    def __init__(self, c=1.0, random_state=RANDOM_STATE):
        self.meta = LogisticRegression(C=c, max_iter=2000, random_state=random_state)
        self._fitted = False

    def fit(self, prob_a_val: np.ndarray, prob_b_val: np.ndarray, y_val: np.ndarray):
        x = np.column_stack([prob_a_val, prob_b_val])
        self.meta.fit(x, y_val)
        self._fitted = True

    def predict_proba(self, prob_a: np.ndarray, prob_b: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("FusionModel must be fitted before predict_proba.")
        return self.meta.predict_proba(np.column_stack([prob_a, prob_b]))[:, 1]
