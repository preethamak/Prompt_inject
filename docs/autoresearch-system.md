# AutoResearch System Architecture

## Purpose

`autoresearch/` converts the notebook-discovered modeling strategy into a reproducible experiment pipeline. The package is small, but it encodes several strong research decisions:

- deterministic splits
- explicit search space
- branch-wise modeling
- validation-constrained threshold calibration
- persistent experiment logging
- post hoc error mining

## End-To-End Flow

The runtime flow is:

1. Load and format the dataset.
2. Split into deterministic train, validation, and test partitions.
3. Train the lexical branch on TF-IDF features.
4. Train the semantic branch on sentence embeddings.
5. Generate out-of-fold branch probabilities on the training split.
6. Fit a fusion meta-model on those out-of-fold predictions.
7. Fit final branch models on the full training split.
8. Select a threshold that maximizes validation F1 under an FPR cap.
9. Evaluate on validation and, if good enough, on test.
10. Save the run result.
11. Export hard errors and slice metrics when needed.

## Configuration Layer

Primary file:

- [`autoresearch/config.py`](../autoresearch/config.py)

### Why the configuration design is good

The config file centralizes assumptions that affect comparability:

- fixed random seed
- dataset size
- text truncation
- split ratios
- threshold policy
- candidate hyperparameter space

This is important because sweep results are only meaningful if different runs remain comparable under the same protocol.

### Main choices

- `RANDOM_STATE = 42`
- `DATASET_SIZE = 10000`
- `CONTEXT_MAX_CHARS = 1500`
- split: `70/15/15`
- threshold constraint: `FPR <= 0.10`
- primary metric: `val_f1`
- baseline gate: `BASELINE_F1 = 0.90`

### Research interpretation

The package treats validation F1 as the main optimization target, but it also encodes a deployment-style constraint through the FPR limit. That is stronger than a pure leaderboard mindset because it acknowledges that false positives matter in security filtering pipelines.

## Data Layer

Primary file:

- [`autoresearch/data.py`](../autoresearch/data.py)

### Text construction

Every example is converted to:

```text
Context: <truncated context>
User intent: <intent>
```

This formatting is consistent across the main experiments. It preserves the two main information channels in the dataset:

- retrieved or observed context
- the explicit user request

### Why truncation exists

`context_max_chars` prevents long contexts from dominating memory and embedding cost. The downside is that late-position attacks may be lost, so this parameter is a tradeoff between efficiency and recall.

### Why deterministic splitting matters

The repository compares multiple models and thresholds. If the split changes each time, performance differences become partly noise. The fixed split is a correct design for controlled iteration.

## Feature Layer

Primary file:

- [`autoresearch/features.py`](../autoresearch/features.py)

## Branch A: TF-IDF Plus Logistic Regression

### Representation

- word n-grams
- character n-grams with `char_wb`
- sparse concatenation with `scipy.sparse.hstack`

### Why this branch exists

This branch is optimized for lexical evidence:

- suspicious phrases
- repeated string fragments
- obfuscated but still local attack patterns
- tabular and templated artifacts that dense embeddings may smooth away

### Why Logistic Regression is a good fit

Given a high-dimensional sparse vector space, linear models are a strong baseline and often very competitive. They are also fast, interpretable at the feature level, and stable under repeated evaluation.

## Branch B: Embeddings Plus XGBoost

### Representation

- sentence-level dense embeddings
- pluggable encoder model IDs
- XGBoost over the dense vectors

### Why this branch exists

This branch is optimized for semantic generalization:

- paraphrases
- indirect intent
- contextual meaning that is not recoverable from exact n-grams alone

### Why XGBoost is paired with embeddings

XGBoost can model non-linear interactions in a dense feature space without requiring a large end-to-end fine-tuning setup. That makes it a practical middle ground between neural fine-tuning and simple linear baselines.

## Embedding Cache Design

Two cache layers are used:

1. in-memory cache for repeated access during one session
2. disk cache in `autoresearch_results/embed_cache/`

### Why this approach is useful

Embedding generation is the dominant cost in many reruns. Caching directly improves iteration speed without changing model behavior.

### Important limitation

The cache key is based on:

- model ID
- number of texts
- a snippet built from the first and last few texts

This is lightweight, but not collision-proof. It is acceptable for a local research workflow, but not ideal for production-grade experiment tracking.

## Fusion Layer

Class:

- `FusionModel`

### How it works

The fusion model trains Logistic Regression on a two-column matrix:

- probability from TF-IDF branch
- probability from embedding branch

### Why fusion is justified

Fusion is not added for complexity alone. It is motivated by complementary errors:

- lexical branch tends to be strong on explicit wording
- embedding branch tends to be stronger on semantic variants

If the error sets differ, a meta-classifier can improve overall ranking and operating-point quality.

### Why out-of-fold fusion is important

The previous version trained fusion directly on validation predictions, which mixed meta-model fitting with model selection. The upgraded protocol instead learns fusion weights from out-of-fold training predictions, then keeps validation for threshold selection and comparison only. That makes the reported validation metrics more trustworthy.

## Evaluation Layer

Primary file:

- [`autoresearch/evaluate.py`](../autoresearch/evaluate.py)

### Metric design

The package computes:

- accuracy
- F1
- precision
- recall
- ROC-AUC
- PR-AUC
- TPR at multiple FPR operating points

### Why this evaluation design is stronger than a single metric

Security classification is operating-point sensitive. Two models with similar F1 can behave very differently under low-FPR constraints. Reporting TPR at bounded FPR values makes the evaluation more deployment-relevant.

### Threshold selection logic

`select_threshold` now searches exact score-derived threshold candidates and keeps the one with best F1 subject to `FPR <= max_fpr`.

### Why this is an important methodological choice

The repository does not treat threshold as an arbitrary reporting knob. It treats threshold as part of the system design, which is the correct approach when risk tolerance matters. The exact-search upgrade removes the approximation error from the older grid-based search.

## Slice Diagnostics

The evaluation layer now computes heuristic slices for:

- short context
- long context
- table-heavy context
- narrative context
- question-like intent
- instruction-like intent

These slices are not domain-perfect labels, but they are stable enough to detect regressions that aggregate accuracy would hide.

## Runner Layer

Primary file:

- [`autoresearch/runner.py`](../autoresearch/runner.py)

### Candidate generation

Two sweep modes exist:

- `small`: four representative configurations
- `medium`: cross-product sweep over the major hyperparameters

### Why the small sweep is useful

It provides a quick controlled benchmark set for iteration. That is practical when embedding generation is expensive.

### Why the medium sweep exists

It expands the search space without changing the surrounding protocol. This separates search effort from evaluation rules, which is good experimental hygiene.

### Test gating

Test metrics are only computed when `val_f1 >= BASELINE_F1`.

### Why this design is defensible

It reduces unnecessary test-set peeking for weak runs. The repository still records the validation behavior for all runs, but reserves detailed test reporting for candidates that are plausibly competitive.

## Persistence And Error Analysis

Primary files:

- [`autoresearch/leaderboard.py`](../autoresearch/leaderboard.py)
- [`autoresearch/error_analysis.py`](../autoresearch/error_analysis.py)

### Result persistence

Each run is appended to `autoresearch_results/runs.jsonl`.

This is simple, readable, versionable, and sufficient for a small experiment framework.

### Error mining

The package exports:

- hard false negatives: attack examples predicted too safely
- hard false positives: benign examples predicted too aggressively

### Why this matters

The strongest improvement loop after metric saturation is often data-centric:

- inspect systematic misses
- identify annotation quirks
- find missing attack styles
- spot spurious lexical triggers

The `errors_s4.json` artifact is the start of that loop.
