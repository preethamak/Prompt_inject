# Prompt Injection Detection Research Repository

This repository studies indirect prompt injection detection on `MAlmasabi/Indirect-Prompt-Injection-BIPIA-GPT`. It contains two kinds of work:

1. Notebook-driven experimentation that established the initial baselines and improvement path.
2. A small Python package, `autoresearch/`, that turns the final approach into a reproducible experiment pipeline.

The codebase converges on a two-branch design:

- A lexical branch: TF-IDF over word and character n-grams with Logistic Regression.
- A semantic branch: sentence embeddings with XGBoost.
- A fusion layer: Logistic Regression over branch probabilities.
- A validation-calibrated threshold with an explicit false-positive-rate cap.
- An out-of-fold fusion protocol so validation stays reserved for model selection.
- A layered defense pass that scores context chunks, removes risky spans, and applies a task-alignment gate.
- Slice-based diagnostics for table-heavy, long-context, and intent-style examples.

The current best recorded run is `s4`, which achieved:

- Validation: `accuracy=0.922667`, `f1=0.923885`
- Test: `accuracy=0.924051`, `f1=0.924303`
- Threshold: `0.4604`

These numbers were produced by the earlier fusion protocol stored in `autoresearch_results/`. The codebase now uses a stricter protocol with out-of-fold fusion training and exact threshold search, so the next sweep should be treated as the new reference benchmark.

Separately, the strongest notebook-only run now recorded in `notebook_results/accuracy_runs.jsonl` is:

- Model: `tfidf_logreg_tuned_plus_dense`
- Validation: `f1=0.935013`
- Test: `accuracy=0.929333`, `f1=0.930079`
- Threshold: `0.5000`

This notebook path is the current fastest way to iterate on accuracy without changing `autoresearch/`.

## Documentation Map

Start here if you want the full explanation set:

- [Documentation hub](docs/README.md)
- [Research narrative and experiment history](docs/experiments.md)
- [Notebook-only accuracy experiments](docs/notebook-accuracy-experiments.md)
- [AutoResearch system architecture](docs/autoresearch-system.md)
- [Module and method reference](docs/module-reference.md)
- [Results, limitations, and recommended corrections](docs/results-and-recommendations.md)

## Repository Layout

- `autoresearch/`: reproducible experiment package
- `autoresearch_results/`: saved sweep results, embedding cache, and hard-error exports
- `AutoResearch.ipynb`: package-driven sweep notebook
- `TF_IDF.ipynb`: baseline-to-fusion development notebook
- `XG_Boost.ipynb`: parallel experimentation notebook with overlapping model progression
- `notebook_utils/`: notebook-side reusable helpers for offline data loading, tuned lexical runs, and experiment logging
- `notebook_results/`: logged notebook-only experiment results
- `scripts/`: runnable entrypoints for single-run, sweep, layered defense, and notebook refresh flows

## Quick Reading Guide

If your goal is:

- To understand the research story, read [docs/experiments.md](docs/experiments.md).
- To improve accuracy without touching `autoresearch/`, read [docs/notebook-accuracy-experiments.md](docs/notebook-accuracy-experiments.md).
- To understand how the final pipeline works, read [docs/autoresearch-system.md](docs/autoresearch-system.md).
- To inspect every class/function and why it exists, read [docs/module-reference.md](docs/module-reference.md).
- To decide what should be improved next, read [docs/results-and-recommendations.md](docs/results-and-recommendations.md).

## Run Paths

- `python scripts/run_s4.py`: current strongest base detector path
- `python scripts/run_routed_s4.py`: low-compute routing between TF-IDF and fusion
- `python scripts/run_layered_s4.py`: strongest detector plus layered defense evaluation
- `python scripts/run_small_sweep.py`: full small sweep with persisted results
- `python scripts/update_autoresearch_notebook.py`: execute and record notebook outputs
- `python scripts/update_accuracy_notebooks.py`: refresh the notebook-only accuracy section in `TF_IDF.ipynb` and `XG_Boost.ipynb`
