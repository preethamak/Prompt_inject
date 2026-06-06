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
- Slice-based diagnostics for table-heavy, long-context, and intent-style examples.

The current best recorded run is `s4`, which achieved:

- Validation: `accuracy=0.922667`, `f1=0.923885`
- Test: `accuracy=0.924051`, `f1=0.924303`
- Threshold: `0.4604`

These numbers were produced by the earlier fusion protocol stored in `autoresearch_results/`. The codebase now uses a stricter protocol with out-of-fold fusion training and exact threshold search, so the next sweep should be treated as the new reference benchmark.

## Documentation Map

Start here if you want the full explanation set:

- [Documentation hub](docs/README.md)
- [Research narrative and experiment history](docs/experiments.md)
- [AutoResearch system architecture](docs/autoresearch-system.md)
- [Module and method reference](docs/module-reference.md)
- [Results, limitations, and recommended corrections](docs/results-and-recommendations.md)

## Repository Layout

- `autoresearch/`: reproducible experiment package
- `autoresearch_results/`: saved sweep results, embedding cache, and hard-error exports
- `AutoResearch.ipynb`: package-driven sweep notebook
- `TF_IDF.ipynb`: baseline-to-fusion development notebook
- `XG_Boost.ipynb`: parallel experimentation notebook with overlapping model progression

## Quick Reading Guide

If your goal is:

- To understand the research story, read [docs/experiments.md](docs/experiments.md).
- To understand how the final pipeline works, read [docs/autoresearch-system.md](docs/autoresearch-system.md).
- To inspect every class/function and why it exists, read [docs/module-reference.md](docs/module-reference.md).
- To decide what should be improved next, read [docs/results-and-recommendations.md](docs/results-and-recommendations.md).
