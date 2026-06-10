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
from .defenses import (
    INSTRUCTION_RX,
    LayeredDefensePipeline,
    build_model_text,
    split_context_into_chunks,
    summarize_defense_assessments,
)
from .routing import build_routed_probabilities, select_routing_strategy


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


def _aggregate_chunk_probabilities(chunk_prob: np.ndarray, top_k: int = 2) -> tuple[float, float]:
    if chunk_prob.size == 0:
        return 0.0, 0.0
    top = np.sort(chunk_prob)[-top_k:]
    return float(np.max(chunk_prob)), float(np.mean(top))


def _build_meta_features_from_rows(
    rows: list[dict],
    tfidf_branch: TfidfBranch,
    xgb_branch: EmbedXGBBranch,
    full_prob_tfidf: np.ndarray,
    full_prob_xgb: np.ndarray,
    context_max_chars: int,
    chunk_chars: int,
    overlap_chars: int,
) -> np.ndarray:
    features: list[list[float]] = []
    row_chunks: list[list[str]] = []
    flat_chunk_texts: list[str] = []

    for row in rows:
        context = str(row.get("context", ""))
        user_intent = str(row.get("user_intent", ""))
        chunks = split_context_into_chunks(
            context,
            chunk_chars=chunk_chars,
            overlap_chars=overlap_chars,
        )
        chunk_texts = [
            build_model_text(chunk_text, user_intent, context_max_chars)
            for _, _, chunk_text in chunks
        ]
        row_chunks.append(chunk_texts)
        flat_chunk_texts.extend(chunk_texts)

    if flat_chunk_texts:
        flat_chunk_prob_tfidf = tfidf_branch.predict_proba(flat_chunk_texts)
        flat_chunk_prob_xgb = xgb_branch.predict_proba(flat_chunk_texts)
    else:
        flat_chunk_prob_tfidf = np.array([], dtype=float)
        flat_chunk_prob_xgb = np.array([], dtype=float)

    cursor = 0
    for row, prob_tfidf, prob_xgb, chunk_texts in zip(rows, full_prob_tfidf, full_prob_xgb, row_chunks):
        n_chunks = len(chunk_texts)
        if n_chunks:
            chunk_prob_tfidf = flat_chunk_prob_tfidf[cursor:cursor + n_chunks]
            chunk_prob_xgb = flat_chunk_prob_xgb[cursor:cursor + n_chunks]
        else:
            chunk_prob_tfidf = np.array([], dtype=float)
            chunk_prob_xgb = np.array([], dtype=float)
        cursor += n_chunks

        context = str(row.get("context", ""))
        user_intent = str(row.get("user_intent", ""))
        chunk_tfidf_max, chunk_tfidf_top2 = _aggregate_chunk_probabilities(chunk_prob_tfidf)
        chunk_xgb_max, chunk_xgb_top2 = _aggregate_chunk_probabilities(chunk_prob_xgb)

        instruction_hits = len(INSTRUCTION_RX.findall(context))
        pipe_count = context.count("|")
        newline_count = context.count("\n")

        features.append([
            float(prob_tfidf),
            float(prob_xgb),
            float(abs(prob_tfidf - prob_xgb)),
            chunk_tfidf_max,
            chunk_tfidf_top2,
            chunk_xgb_max,
            chunk_xgb_top2,
            float(np.log1p(len(context))),
            float(np.log1p(len(user_intent))),
            float(np.log1p(n_chunks)),
            float(np.log1p(instruction_hits)),
            float(np.log1p(pipe_count)),
            float(np.log1p(newline_count)),
        ])

    return np.asarray(features, dtype=float)


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
    oof_meta = np.zeros((len(split.x_train), 13), dtype=float)
    x_train = np.array(split.x_train, dtype=object)
    y_train = np.asarray(split.y_train)
    rows_train = np.array(split.rows_train, dtype=object)
    train_embeddings = _get_embeddings(cfg.embed_model, split.x_train)

    for train_idx, holdout_idx in skf.split(x_train, y_train):
        fold_x_train = x_train[train_idx].tolist()
        fold_y_train = y_train[train_idx]
        fold_x_holdout = x_train[holdout_idx].tolist()
        fold_rows_holdout = rows_train[holdout_idx].tolist()

        tfidf = _build_tfidf_branch(cfg)
        tfidf.fit_transform(fold_x_train, fold_y_train)
        holdout_prob_tfidf = tfidf.predict_proba(fold_x_holdout)

        xgb_branch = _build_xgb_branch(cfg)
        xgb_branch.clf.fit(train_embeddings[train_idx], fold_y_train)
        holdout_prob_xgb = xgb_branch.clf.predict_proba(
            train_embeddings[holdout_idx]
        )[:, 1]
        oof_meta[holdout_idx] = _build_meta_features_from_rows(
            rows=fold_rows_holdout,
            tfidf_branch=tfidf,
            xgb_branch=xgb_branch,
            full_prob_tfidf=holdout_prob_tfidf,
            full_prob_xgb=holdout_prob_xgb,
            context_max_chars=cfg.context_max_chars,
            chunk_chars=cfg.defense_chunk_chars,
            overlap_chars=cfg.defense_overlap_chars,
        )

    fusion = FusionModel(c=cfg.fusion_c)
    fusion.fit(oof_meta, y_train)
    return fusion, oof_meta


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

        val_meta = _build_meta_features_from_rows(
            rows=split.rows_val,
            tfidf_branch=tfidf,
            xgb_branch=xgb_branch,
            full_prob_tfidf=val_prob_tfidf,
            full_prob_xgb=val_prob_xgb,
            context_max_chars=cfg.context_max_chars,
            chunk_chars=cfg.defense_chunk_chars,
            overlap_chars=cfg.defense_overlap_chars,
        )
        test_meta = _build_meta_features_from_rows(
            rows=split.rows_test,
            tfidf_branch=tfidf,
            xgb_branch=xgb_branch,
            full_prob_tfidf=test_prob_tfidf,
            full_prob_xgb=test_prob_xgb,
            context_max_chars=cfg.context_max_chars,
            chunk_chars=cfg.defense_chunk_chars,
            overlap_chars=cfg.defense_overlap_chars,
        )

        val_prob_fusion  = fusion.predict_proba(val_meta)
        test_prob_fusion = fusion.predict_proba(test_meta)

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

        thr_a, _ = select_threshold(split.y_val, val_prob_tfidf, max_fpr=cfg.max_fpr)
        routing = select_routing_strategy(
            y_true=split.y_val,
            tfidf_prob=val_prob_tfidf,
            fusion_prob=val_prob_fusion,
            tfidf_threshold=thr_a,
            max_fpr=cfg.max_fpr,
            margin_grid=cfg.routing_margin_grid,
        )
        val_prob_routed, val_use_fusion = build_routed_probabilities(
            tfidf_prob=val_prob_tfidf,
            fusion_prob=val_prob_fusion,
            tfidf_threshold=routing.tfidf_threshold,
            margin=routing.margin,
        )
        result.update(evaluate_at_threshold(
            split.y_val, val_prob_routed, routing.final_threshold, prefix="val_routed"
        ))
        result["routing_margin"] = round(routing.margin, 4)
        result["routing_tfidf_threshold"] = round(routing.tfidf_threshold, 4)
        result["routing_final_threshold"] = round(routing.final_threshold, 4)
        result["routing_val_f1"] = round(routing.val_f1, 6)
        result["routing_val_used_fusion_fraction"] = round(routing.used_fusion_fraction, 6)
        result["routing_val_used_tfidf_fraction"] = round(1.0 - routing.used_fusion_fraction, 6)

        if cfg.enable_defense:
            defense_threshold = min(0.99, thr + cfg.defense_chunk_threshold_delta)
            result["defense_chunk_threshold"] = round(defense_threshold, 4)
            defense = LayeredDefensePipeline(
                tfidf_branch=tfidf,
                xgb_branch=xgb_branch,
                fusion_model=fusion,
                context_max_chars=cfg.context_max_chars,
                chunk_threshold=defense_threshold,
                chunk_chars=cfg.defense_chunk_chars,
                overlap_chars=cfg.defense_overlap_chars,
            )
            val_assessments = defense.analyze_rows(split.rows_val)
            val_sanitized_rows = [
                {"context": a.sanitized_context, "user_intent": a.user_intent}
                for a in val_assessments
            ]
            val_sanitized_tfidf = tfidf.predict_proba([a.sanitized_text for a in val_assessments])
            val_sanitized_xgb = xgb_branch.predict_proba([a.sanitized_text for a in val_assessments])
            val_prob_sanitized = fusion.predict_proba(_build_meta_features_from_rows(
                rows=val_sanitized_rows,
                tfidf_branch=tfidf,
                xgb_branch=xgb_branch,
                full_prob_tfidf=val_sanitized_tfidf,
                full_prob_xgb=val_sanitized_xgb,
                context_max_chars=cfg.context_max_chars,
                chunk_chars=cfg.defense_chunk_chars,
                overlap_chars=cfg.defense_overlap_chars,
            ))
            result.update(evaluate_at_threshold(
                split.y_val, val_prob_sanitized, thr, prefix="val_sanitized"
            ))
            result.update(summarize_defense_assessments(
                val_assessments, prefix="val_defense"
            ))

        # ── Test metrics only if val_f1 beats baseline ────────────────────────
        beats_baseline = val_metrics.get("val_f1", 0.0) >= BASELINE_F1
        result["beats_baseline"] = beats_baseline
        if beats_baseline:
            test_prob_routed, test_use_fusion = build_routed_probabilities(
                tfidf_prob=test_prob_tfidf,
                fusion_prob=test_prob_fusion,
                tfidf_threshold=routing.tfidf_threshold,
                margin=routing.margin,
            )
            test_metrics = evaluate_at_threshold(
                split.y_test, test_prob_fusion, thr, prefix="test"
            )
            result.update(test_metrics)
            result.update(evaluate_at_threshold(
                split.y_test, test_prob_routed, routing.final_threshold, prefix="test_routed"
            ))
            result["routing_test_used_fusion_fraction"] = round(float(np.mean(test_use_fusion)), 6)
            result["routing_test_used_tfidf_fraction"] = round(float(1.0 - np.mean(test_use_fusion)), 6)
            result.update(compute_slice_metrics(
                split.y_test, test_prob_fusion, split.rows_test, thr, prefix="test_slice"
            ))
            if cfg.enable_defense:
                test_assessments = defense.analyze_rows(split.rows_test)
                test_sanitized_rows = [
                    {"context": a.sanitized_context, "user_intent": a.user_intent}
                    for a in test_assessments
                ]
                test_sanitized_tfidf = tfidf.predict_proba([a.sanitized_text for a in test_assessments])
                test_sanitized_xgb = xgb_branch.predict_proba([a.sanitized_text for a in test_assessments])
                test_prob_sanitized = fusion.predict_proba(_build_meta_features_from_rows(
                    rows=test_sanitized_rows,
                    tfidf_branch=tfidf,
                    xgb_branch=xgb_branch,
                    full_prob_tfidf=test_sanitized_tfidf,
                    full_prob_xgb=test_sanitized_xgb,
                    context_max_chars=cfg.context_max_chars,
                    chunk_chars=cfg.defense_chunk_chars,
                    overlap_chars=cfg.defense_overlap_chars,
                ))
                result.update(evaluate_at_threshold(
                    split.y_test, test_prob_sanitized, thr, prefix="test_sanitized"
                ))
                result.update(summarize_defense_assessments(
                    test_assessments, prefix="test_defense"
                ))

        # ── Also log individual branch val metrics for diagnosis ──────────────
        thr_b, _ = select_threshold(split.y_val, val_prob_xgb, max_fpr=cfg.max_fpr)
        result.update(evaluate_at_threshold(split.y_val, val_prob_tfidf, thr_a, prefix="val_tfidf"))
        result.update(evaluate_at_threshold(split.y_val, val_prob_xgb, thr_b, prefix="val_xgb"))
        result["fusion_meta_features"] = int(oof_stack.shape[1])
        if cfg.enable_defense:
            result["protocol"] = "oof_chunk_meta_fusion_routing_exact_threshold_layered_defense_v4"
        else:
            result["protocol"] = "oof_chunk_meta_fusion_routing_exact_threshold_v4"

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
