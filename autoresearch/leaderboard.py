"""
Results persistence and leaderboard utilities.

Each run is appended as one JSON line to results/runs.jsonl.
Leaderboard is sorted by val_f1 (primary) then tpr@0.1fpr (secondary).
"""

from __future__ import annotations
import json, os
from pathlib import Path
import pandas as pd

RESULTS_DIR = Path(__file__).parent.parent / "autoresearch_results"
RUNS_FILE = RESULTS_DIR / "runs.jsonl"


def _ensure_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_result(result: dict):
    """Append one run result to the JSONL results file."""
    _ensure_dir()
    with RUNS_FILE.open("a") as f:
        f.write(json.dumps(result) + "\n")


def load_results() -> list[dict]:
    """Load all results from the JSONL file."""
    if not RUNS_FILE.exists():
        return []
    rows = []
    with RUNS_FILE.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def clear_results():
    """Delete the results file (use before re-running a full sweep)."""
    if RUNS_FILE.exists():
        RUNS_FILE.unlink()
        print(f"Cleared {RUNS_FILE}")


def get_leaderboard(top_n: int = 20, beats_baseline_only: bool = False) -> pd.DataFrame:
    """
    Return a ranked DataFrame of results.
    Primary sort: val_f1 descending.
    Secondary sort: tpr@0.1fpr descending.
    """
    rows = load_results()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)

    if beats_baseline_only and "beats_baseline" in df.columns:
        df = df[df["beats_baseline"] == True]

    sort_cols = []
    if "val_f1" in df.columns:
        sort_cols.append("val_f1")
    tpr_col = "val_tpr_at_fpr_0.1000"
    if tpr_col in df.columns:
        sort_cols.append(tpr_col)

    if sort_cols:
        df = df.sort_values(sort_cols, ascending=False).reset_index(drop=True)

    return df.head(top_n)


def display_leaderboard(top_n: int = 10, beats_baseline_only: bool = False):
    """Pretty-print the leaderboard. Call from a notebook cell."""
    df = get_leaderboard(top_n=top_n, beats_baseline_only=beats_baseline_only)
    if df.empty:
        print("No results yet.")
        return

    show_cols = [
        "run_id", "embed_model",
        "word_ngram", "char_ngram", "logreg_c", "fusion_c",
        "val_f1", "val_accuracy", "val_roc_auc",
        "val_tpr_at_fpr_0.1000", "val_tpr_at_fpr_0.0100",
        "selected_threshold", "beats_baseline", "elapsed_s",
    ]
    show_cols = [c for c in show_cols if c in df.columns]

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.4f}".format)

    print(f"=== AutoResearch Leaderboard (top {min(top_n, len(df))}) ===")
    print(df[show_cols].to_string(index=True))


def get_best_run() -> dict | None:
    """Return the single best run dict by val_f1."""
    rows = load_results()
    if not rows:
        return None
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if not ok_rows:
        return None
    return max(ok_rows, key=lambda r: r.get("val_f1", 0.0))
