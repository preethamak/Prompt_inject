"""
AutoResearch sweep runner.

generate_candidates() produces a list of RunConfig objects covering
the full small search space.  run_single() trains and evaluates one config.
run_sweep() iterates the full list, logs each result, and returns a results list.

Test metrics are computed only for runs that beat BASELINE_F1 on val.
"""

from __future__ import annotations
import itertools, time, traceback
from typing import Callable
import numpy as np
from sklearn.model_selection import StratifiedKFold

from .config import (
    RunConfig, RANDOM_STATE, BASELINE_F1,
    TFIDF_WORD_NGRAMS, TFIDF_CHAR_NGRAMS,
    LOGREG_C_VALUES, EMBED_MODELS, XGB_CONFIGS, FUSION_META_C,
)
from .data import DataSplit
from .features import TfidfBranch, EmbedXGBBranch, FusionModel, _get_embeddings
from .evaluate import select_threshold, evaluate_at_threshold, compute_slice_metrics


def generate_candidates(search_size: str = "small") -> list[RunConfig]:
    """
    Return list of RunConfig candidates.
    search_size='small'  — 4 carefully chosen representative configs
    search_size='medium' — full cross-product of search space
    """
    if search_size == "small":
        return [
            RunConfig(run_id="s1", word_ngram=(1, 2), char_ngram=(3, 5),
                      logreg_c=4.0, embed_model=EMBED_MODELS[0], xgb=XGB_CONFIGS[0], fusion_c=1.0),
            RunConfig(run_id="s2", word_ngram=(1, 2), char_ngram=(3, 5),
                      logreg_c=4.0, embed_model=EMBED_MODELS[1], xgb=XGB_CONFIGS[0], fusion_c=1.0),
            RunConfig(run_id="s3", word_ngram=(1, 3), char_ngram=(3, 6),
                      logreg_c=8.0, embed_model=EMBED_MODELS[0], xgb=XGB_CONFIGS[1], fusion_c=2.0),
            RunConfig(run_id="s4", word_ngram=(1, 3), char_ngram=(3, 6),
                      logreg_c=8.0, embed_model=EMBED_MODELS[1], xgb=XGB_CONFIGS[1], fusion_c=2.0),
        ]

    # medium: cross-product of key axes
    candidates = []
    combos = list(itertools.product(
        TFIDF_WORD_NGRAMS, TFIDF_CHAR_NGRAMS,
        LOGREG_C_VALUES, EMBED_MODELS,
        range(len(XGB_CONFIGS)), FUSION_META_C,
    ))
    for idx, (wng, cng, lc, em, xi, fc) in enumerate(combos):
        candidates.append(RunConfig(
            run_id=f"m{idx:04d}",
            word_ngram=wng, char_ngram=cng,
            logreg_c=lc, embed_model=em,
            xgb=XGB_CONFIGS[xi], fusion_c=fc,
        ))
    return candidates


def _build_tfidf_branch(cfg: RunConfig) -> TfidfBranch:
    return TfidfBranch(
        word_ngram=cfg.word_ngram,
        char_ngram=cfg.char_ngram,
        max_word_features=cfg.tfidf_max_word_features,
        max_char_features=cfg.tfidf_max_char_features,
        logreg_c=cfg.logreg_c,
    )


def _build_xgb_branch(cfg: RunConfig) -> EmbedXGBBranch:
    return EmbedXGBBranch(model_id=cfg.embed_model, xgb_params=cfg.xgb)


def _fit_fusion_oof(cfg: RunConfig, split: DataSplit) -> tuple[FusionModel, np.ndarray]:
    """
    Fit fusion on out-of-fold branch predictions from the training split only.
    This keeps validation reserved for threshold selection and model comparison.
    """
    skf = StratifiedKFold(
        n_splits=cfg.fusion_oof_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    oof_prob_tfidf = np.zeros(len(split.x_train), dtype=float)
    oof_prob_xgb = np.zeros(len(split.x_train), dtype=float)

    x_train = np.array(split.x_train, dtype=object)
    y_train = np.asarray(split.y_train)
    train_embeddings = _get_embeddings(cfg.embed_model, split.x_train)

    for train_idx, holdout_idx in skf.split(x_train, y_train):
        fold_x_train = x_train[train_idx].tolist()
        fold_y_train = y_train[train_idx]
        fold_x_holdout = x_train[holdout_idx].tolist()

        tfidf = _build_tfidf_branch(cfg)
        tfidf.fit_transform(fold_x_train, fold_y_train)
        oof_prob_tfidf[holdout_idx] = tfidf.predict_proba(fold_x_holdout)

        xgb_branch = _build_xgb_branch(cfg)
        xgb_branch.clf.fit(train_embeddings[train_idx], fold_y_train)
        oof_prob_xgb[holdout_idx] = xgb_branch.clf.predict_proba(
            train_embeddings[holdout_idx]
        )[:, 1]

    fusion = FusionModel(c=cfg.fusion_c)
    fusion.fit(oof_prob_tfidf, oof_prob_xgb, y_train)
    return fusion, np.column_stack([oof_prob_tfidf, oof_prob_xgb])


def run_single(cfg: RunConfig, split: DataSplit) -> dict:
    """
    Train TF-IDF, XGBoost, and Fusion models for one RunConfig.
    Returns a result dict with config + val metrics + (optional) test metrics.
    """
    t0 = time.time()
    result = cfg.as_dict()
    result["status"] = "ok"
    result["error"] = ""

    try:
        # ── Fusion training via out-of-fold branch predictions ───────────────
        fusion, oof_stack = _fit_fusion_oof(cfg, split)
        result["fusion_train_oof_rows"] = int(oof_stack.shape[0])

        # ── Branch A: TF-IDF on full train ───────────────────────────────────
        tfidf = _build_tfidf_branch(cfg)
        tfidf.fit_transform(split.x_train, split.y_train)

        val_prob_tfidf  = tfidf.predict_proba(split.x_val)
        test_prob_tfidf = tfidf.predict_proba(split.x_test)

        # ── Branch B: Embed + XGBoost on full train ──────────────────────────
        xgb_branch = _build_xgb_branch(cfg)
        xgb_branch.fit(split.x_train, split.y_train)

        val_prob_xgb  = xgb_branch.predict_proba(split.x_val)
        test_prob_xgb = xgb_branch.predict_proba(split.x_test)

        val_prob_fusion  = fusion.predict_proba(val_prob_tfidf, val_prob_xgb)
        test_prob_fusion = fusion.predict_proba(test_prob_tfidf, test_prob_xgb)

        # ── Select threshold on val (primary objective: val_f1 @ FPR<=max_fpr)
        thr, val_f1_calibrated = select_threshold(
            split.y_val, val_prob_fusion, max_fpr=cfg.max_fpr
        )

        val_metrics = evaluate_at_threshold(
            split.y_val, val_prob_fusion, thr, prefix="val"
        )
        result.update(val_metrics)
        result.update(compute_slice_metrics(
            split.y_val, val_prob_fusion, split.rows_val, thr, prefix="val_slice"
        ))
        result["val_f1_calibrated"] = round(val_f1_calibrated, 6)
        result["selected_threshold"] = round(thr, 4)

        # ── Test metrics only if val_f1 beats baseline ────────────────────────
        beats_baseline = val_metrics.get("val_f1", 0.0) >= BASELINE_F1
        result["beats_baseline"] = beats_baseline
        if beats_baseline:
            test_metrics = evaluate_at_threshold(
                split.y_test, test_prob_fusion, thr, prefix="test"
            )
            result.update(test_metrics)
            result.update(compute_slice_metrics(
                split.y_test, test_prob_fusion, split.rows_test, thr, prefix="test_slice"
            ))

        # ── Also log individual branch val metrics for diagnosis ──────────────
        thr_a, _ = select_threshold(split.y_val, val_prob_tfidf, max_fpr=cfg.max_fpr)
        thr_b, _ = select_threshold(split.y_val, val_prob_xgb, max_fpr=cfg.max_fpr)
        result.update(evaluate_at_threshold(split.y_val, val_prob_tfidf, thr_a, prefix="val_tfidf"))
        result.update(evaluate_at_threshold(split.y_val, val_prob_xgb, thr_b, prefix="val_xgb"))
        result["protocol"] = "oof_fusion_exact_threshold_v1"

    except Exception as exc:
        result["status"] = "error"
        result["error"] = traceback.format_exc()

    result["elapsed_s"] = round(time.time() - t0, 1)
    return result


def run_sweep(
    split: DataSplit,
    candidates: list[RunConfig] | None = None,
    search_size: str = "small",
    on_result: Callable[[dict], None] | None = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Run all candidates and collect results.
    on_result is called after each run (e.g. to persist immediately).
    """
    if candidates is None:
        candidates = generate_candidates(search_size)

    results = []
    n = len(candidates)
    for i, cfg in enumerate(candidates):
        if verbose:
            print(f"\n[{i+1}/{n}] run_id={cfg.run_id}  "
                  f"embed={cfg.embed_model.split('/')[-1]}  "
                  f"word_ng={cfg.word_ngram}  char_ng={cfg.char_ngram}  "
                  f"C={cfg.logreg_c}  fusion_c={cfg.fusion_c}")
        res = run_single(cfg, split)
        results.append(res)
        if verbose:
            status = res.get("status", "?")
            vf1 = res.get("val_f1", "n/a")
            thr = res.get("selected_threshold", "n/a")
            tpr = res.get("val_tpr_at_fpr_0.1000", "n/a")
            print(f"    status={status}  val_f1={vf1}  thr={thr}  "
                  f"tpr@0.1fpr={tpr}  time={res.get('elapsed_s', '?')}s")
        if on_result:
            on_result(res)

    return results
