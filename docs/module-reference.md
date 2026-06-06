# Module And Method Reference

This document maps each implemented class, function, and notebook section to its role in the system.

## Package Overview

Package:

- [`autoresearch/`](../autoresearch)

Modules:

- [`__init__.py`](../autoresearch/__init__.py)
- [`config.py`](../autoresearch/config.py)
- [`data.py`](../autoresearch/data.py)
- [`features.py`](../autoresearch/features.py)
- [`evaluate.py`](../autoresearch/evaluate.py)
- [`leaderboard.py`](../autoresearch/leaderboard.py)
- [`error_analysis.py`](../autoresearch/error_analysis.py)
- [`runner.py`](../autoresearch/runner.py)

## `autoresearch.__init__`

### Purpose

Declares the package and identifies its scope as prompt injection detection experiments.

### Why it exists

Minimal package initialization keeps imports clean for notebooks and local scripts.

## `autoresearch.config`

### Constants

#### `RANDOM_STATE`

Controls deterministic splitting and model reproducibility where supported.

#### `DATASET_SIZE`

Caps the loaded dataset slice at `10000` examples. This makes experimentation tractable while still large enough to show scale effects.

#### `CONTEXT_MAX_CHARS`

Limits the context length used when building model input text. The repository assumes the most useful signal usually appears early enough to justify truncation.

#### `TRAIN_FRAC`, `VAL_FRAC`

Encode the `70/15/15` split policy.

#### `DATASET_ID`

Pins the dataset source to `MAlmasabi/Indirect-Prompt-Injection-BIPIA-GPT`.

#### `MAX_FPR_FOR_THRESHOLD`

Defines the operating constraint for threshold search. This is one of the most important methodological constants in the repository.

#### `FPR_OPERATING_POINTS`

Requests reporting at several false-positive-rate targets, which makes leaderboard comparisons more informative than using F1 alone.

#### `PRIMARY_METRIC`

Locks the main sweep objective to `val_f1`.

#### `BASELINE_F1`

Acts as a gate for whether test metrics should be computed for a run.

#### `FUSION_OOF_FOLDS`

Defines how many stratified folds are used to generate out-of-fold branch probabilities for fusion training.

### Search space lists

#### `TFIDF_WORD_NGRAMS`, `TFIDF_CHAR_NGRAMS`

These define the lexical receptive field. Wider ranges can capture richer attack wording, but also raise feature dimensionality.

#### `LOGREG_C_VALUES`

Controls regularization strength for the lexical branch.

#### `EMBED_MODELS`

Specifies the sentence encoders compared in the sweep:

- `sentence-transformers/all-MiniLM-L6-v2`
- `BAAI/bge-small-en-v1.5`

#### `XGB_CONFIGS`

A small set of curated XGBoost parameter choices rather than a full unrestricted search.

#### `FUSION_META_C`

Regularization choices for the fusion meta-classifier.

### `RunConfig`

Dataclass that packages one experiment candidate into a single object.

#### Why the dataclass approach is good

It keeps the runner interface clear and makes each run serializable.

#### `as_dict()`

Flattens configuration values into a JSON-friendly dictionary for persistence. XGBoost parameters are expanded into `xgb_*` fields so result rows stay flat and tabular.

## `autoresearch.data`

### `DataSplit`

Dataclass holding train, validation, and test texts and labels.

#### Why this wrapper helps

It keeps split state explicit and prevents accidental mismatching of text arrays, labels, and raw rows.

#### `rows_val`, `rows_test`

These raw rows support slice diagnostics and hard-example exports without rebuilding dataset context later.

#### `sizes`

Convenience property for human-readable split counts.

### `_build_text(row, context_max_chars)`

Constructs the canonical text representation from raw dataset fields.

#### Why this helper matters

It ensures every model branch consumes the same textual framing.

### `load_split(...)`

Loads the dataset and returns a deterministic stratified split.

#### Why this is a key method

It is the point where experimental reproducibility is enforced. Any documentation or future extension should treat this function as part of the protocol, not a disposable utility.

## `autoresearch.features`

### `TfidfBranch`

Encapsulates the lexical branch.

#### `__init__(...)`

Builds:

- a word-level `TfidfVectorizer`
- a character-level `TfidfVectorizer`
- a `LogisticRegression` classifier

Why this design:

- word n-grams capture phrase-level structure
- character n-grams improve robustness to local variation and formatting artifacts
- Logistic Regression is a strong sparse baseline

#### `fit_transform(x_train, y_train)`

Fits vectorizers, stacks the sparse matrices, trains the classifier, and returns the training matrix.

Why this method exists:

It combines fitting and transformation because the branch is only useful after both vectorizers and classifier are trained on aligned features.

#### `transform(texts)`

Transforms new text with the already-fitted vectorizers.

#### `predict_proba(texts)`

Returns class-1 probabilities. The entire fusion system depends on probability output rather than hard labels.

### `_cache_key(model_id, texts)`

Builds a lightweight cache key for embedding arrays.

#### Why it is useful

It avoids recomputation across repeated notebook or sweep runs.

#### What should be improved

It is still only a heuristic identity function, not a full content hash of all inputs.

### `_get_embeddings(model_id, texts, batch_size=64)`

Implements the memory-cache to disk-cache to fresh-encode fallback chain.

#### Why this method is high leverage

The sweep would be much slower without embedding reuse. This function is a performance feature, not just a convenience helper.

### `clear_embed_cache(disk=False)`

Clears memory cache and, optionally, the persisted cache directory.

#### Why it exists

Useful when changing experiments, invalidating stale embeddings, or benchmarking clean runs.

### `EmbedXGBBranch`

Encapsulates the semantic branch.

#### `__init__(model_id, xgb_params, random_state=...)`

Initializes the XGBoost classifier and injects deterministic settings where possible.

#### `fit(x_train, y_train, x_val=None)`

Encodes train texts and fits XGBoost.

Note:

`x_val` is accepted but not used. That makes the interface slightly misleading and is a correction candidate.

#### `predict_proba(texts)`

Encodes input texts and returns positive-class probabilities.

### `FusionModel`

Encapsulates the meta-classifier over branch probabilities.

#### `__init__(c=1.0, random_state=...)`

Creates a Logistic Regression meta-model.

#### `fit(prob_a_val, prob_b_val, y_val)`

Fits the meta-model on validation outputs.

#### Why out-of-fold training is used here

The current runner trains fusion on out-of-fold predictions from the training split. That is a lightweight stacking strategy that preserves validation for model comparison and threshold selection.

#### `predict_proba(prob_a, prob_b)`

Returns fused class-1 probabilities and raises if the model has not been fitted.

## `autoresearch.evaluate`

### `compute_metrics(y_true, y_pred, y_prob, prefix="")`

Computes standard classification metrics and prefixes field names for structured logging.

### `compute_tpr_at_fpr(y_true, y_prob, fpr_targets=..., prefix="")`

Uses the ROC curve to estimate recall at several FPR budgets.

#### Why this is important

It aligns evaluation with practical operating constraints rather than only global ranking summaries.

### `select_threshold(y_true, y_prob, max_fpr=...)`

Searches thresholds and returns the best threshold under the FPR cap.

#### Why the method is strong

It formalizes a deployment policy and makes that policy reproducible.

The implementation now searches exact score-derived thresholds rather than a coarse fixed grid.

### `compute_slice_metrics(y_true, y_prob, rows, threshold, prefix="")`

Computes per-slice metrics for deployment-relevant subsets such as long contexts and table-heavy rows.

#### Why this method matters

It turns error analysis into something comparable across runs instead of a purely qualitative notebook exercise.

### `evaluate_at_threshold(y_true, y_prob, threshold, prefix="")`

Produces the full metric set at one selected operating point.

## `autoresearch.leaderboard`

### `RESULTS_DIR`, `RUNS_FILE`

Centralize experiment artifact locations.

### `_ensure_dir()`

Creates the result directory before writes.

### `save_result(result)`

Appends one JSON object per line.

#### Why JSONL is a good choice here

It is easy to append, easy to inspect, and simple to load into pandas later.

### `load_results()`

Reads all saved run rows.

### `clear_results()`

Deletes the runs file.

### `get_leaderboard(top_n=20, beats_baseline_only=False)`

Loads rows into pandas and sorts by validation F1, then by `val_tpr_at_fpr_0.1000`.

#### Why the secondary sort is sensible

If two runs are close on F1, higher recall at the allowed FPR is a meaningful tiebreaker.

### `display_leaderboard(top_n=10, beats_baseline_only=False)`

Prints a human-readable leaderboard for notebooks.

### `get_best_run()`

Returns the single best completed run according to validation F1.

## `autoresearch.error_analysis`

### `get_hard_false_negatives(...)`

Finds attack examples missed with high confidence.

#### Why this method matters

These cases are often the best candidates for data augmentation or feature redesign.

### `get_hard_false_positives(...)`

Finds benign examples incorrectly flagged with high confidence.

#### Why this method matters

These cases expose spurious lexical triggers and over-aggressive operating points.

### `export_hard_examples(...)`

Writes both hard-error sets plus slice metrics to a JSON file per run.

## `autoresearch.runner`

### `generate_candidates(search_size="small")`

Creates either:

- a curated four-run sweep
- a full cross-product sweep

#### Why this split is practical

It supports both fast iteration and broader search without duplicating evaluation logic.

### `_build_tfidf_branch(cfg)` and `_build_xgb_branch(cfg)`

Small factory helpers so the full-train and fold-level branch instances stay consistent.

### `_fit_fusion_oof(cfg, split)`

Generates out-of-fold branch probabilities on the training split and fits the fusion model on them.

#### Why this method matters

This is the main protocol upgrade in the repository. It removes the earlier leakage between fusion fitting and validation reporting.

### `run_single(cfg, split)`

This is the core experimental method. It performs:

1. lexical branch training
2. embedding branch training
3. out-of-fold fusion training
4. final branch fitting on full training data
5. exact threshold selection on validation
6. validation evaluation plus slice metrics
7. optional test evaluation plus slice metrics
8. branch-level diagnostic logging

#### Why this method is central

It expresses the full modeling hypothesis in one place and defines exactly what "a run" means in this repository.

### `run_sweep(...)`

Iterates candidates, prints progress, collects results, and optionally persists each row through `on_result`.

#### Why the callback design is good

It separates run execution from persistence policy. That keeps the runner reusable in notebooks and scripts.

## Notebook Reference

## `TF_IDF.ipynb`

### Role

This is the most complete exploratory notebook. Despite the filename, it includes:

- imported DeBERTa baseline
- threshold tuning
- embedding plus XGBoost
- BGE comparison
- 10k scale-up
- ensemble experiment
- lexical plus semantic fusion pipeline

### Why it matters

It contains the clearest evolution from failed baseline to strong hybrid model.

## `XG_Boost.ipynb`

### Role

Largely overlaps with the earlier experimental stages:

- baseline inference
- threshold tuning
- embedding plus XGBoost
- BGE comparison
- 10k scale-up
- ensemble test

### Why it still matters

It preserves a parallel experimentation trail, but functionally it duplicates much of `TF_IDF.ipynb`.

### What should be improved

This notebook contains a literal `HF_TOKEN = "API_KEY"` placeholder pattern, which is weaker than the environment-variable approach used in `TF_IDF.ipynb`. The environment-based approach should be standardized.

## `AutoResearch.ipynb`

### Role

Acts as the notebook front end to the `autoresearch/` package:

- loads deterministic splits
- runs the configured sweep
- displays the leaderboard
- compares the best run against baseline expectations
- exports hard errors

### Why it matters

This notebook is the bridge between exploratory work and repeatable experimentation.
