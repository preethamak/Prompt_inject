# Notebook Accuracy Experiments

This file tracks the notebook-side accuracy work that intentionally stays outside `autoresearch/`.

## What Was Added

- `notebook_utils/accuracy_lab.py`
  - cached JSONL dataset loader
  - deterministic `train/val/test` split
  - tuned TF-IDF + Logistic Regression runs
  - optional dense context features
  - optional local BGE + XGBoost branch
  - automatic JSONL experiment logging
- `scripts/update_accuracy_notebooks.py`
  - appends the new accuracy section into `TF_IDF.ipynb` and `XG_Boost.ipynb`
- notebook results log:
  - `notebook_results/accuracy_runs.jsonl`

## Why This Path

The notebooks already contained a stronger classical ML direction, but it was fragmented across exploratory cells and did not leave a clean experiment trail. The new notebook-only helper consolidates that work without changing the `autoresearch/` package.

## Experiment Protocol

Each notebook run now records:

- model name
- validation-selected threshold
- test accuracy
- test F1
- precision / recall
- ROC-AUC / PR-AUC
- dataset limit
- context truncation setting
- timestamp
- model notes

## Default Notebook Runs

The appended section compares these notebook-side candidates:

- `tfidf_logreg_baseline`
- `tfidf_logreg_tuned`
- `tfidf_logreg_tuned_plus_dense`
- `bge_small_xgb_tuned` if the local BGE cache is available

## Recommended Usage

1. Run the new section in `TF_IDF.ipynb` first.
2. Keep `NOTEBOOK_FAST_MODE = True` for quick sanity checks.
3. Switch to `NOTEBOOK_FAST_MODE = False` for the full run you want to keep.
4. Review `notebook_results/accuracy_runs.jsonl` after each full run and keep the best experiment tag as the reference notebook result.

## Notes

- The dataset loader is cache-first and does not depend on a live network connection.
- The embedding branch is optional because the lexical notebook path is faster to iterate on.
- If the embedding branch errors, the run is still logged with `status=error` so failures are traceable instead of disappearing.

## Verification Run Recorded In This Change

Smoke test executed through `run_accuracy_suite(dataset_limit=300, fast_mode=True, include_embeddings=False, experiment_tag="smoke_test")`.

Observed result:

- best model: `tfidf_logreg_tuned_plus_dense`
- test accuracy: `0.755556`
- test F1: `0.784314`
- validation F1: `0.793103`

This was only a fast sanity check to verify the new notebook path and logging behavior. The intended reference run is still the full 10k notebook execution.

## Latest Full Notebook Run

The current full reference run in `notebook_results/accuracy_runs.jsonl` under `experiment_tag="notebook_accuracy_upgrade_v1"` is:

- `tfidf_logreg_baseline`
  - test accuracy: `0.911333`
  - test F1: `0.912211`
- `tfidf_logreg_tuned`
  - test accuracy: `0.918667`
  - test F1: `0.919842`
- `tfidf_logreg_tuned_plus_dense`
  - validation F1: `0.935013`
  - test accuracy: `0.929333`
  - test F1: `0.930079`
  - test ROC-AUC: `0.979398`
  - test PR-AUC: `0.976407`
- `bge_small_xgb_tuned`
  - test accuracy: `0.863333`
  - test F1: `0.870006`

Current notebook conclusion:

- the tuned lexical model improves on the earlier notebook lexical baseline
- adding the lightweight dense structural features helps further
- the standalone BGE + XGBoost branch is weaker than the tuned lexical notebook path
