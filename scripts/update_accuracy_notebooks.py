#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REMOVE_MARKERS = (
    "## Local Accuracy Upgrade Pipeline (TF-IDF + Embeddings + Fusion)",
)


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip("\n").split("\n")],
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip("\n").split("\n")],
    }


SECTION_TAG = "## Notebook Accuracy Upgrade"


def append_or_replace_section(notebook_path: Path, cells_to_append: list[dict]) -> None:
    notebook = json.loads(notebook_path.read_text())
    cells = notebook.get("cells", [])

    filtered_cells: list[dict] = []
    skip_old_block = False
    for cell in cells:
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "markdown" and any(marker in source for marker in REMOVE_MARKERS):
            skip_old_block = True
            continue
        if skip_old_block:
            if cell.get("cell_type") == "markdown" and SECTION_TAG in source:
                skip_old_block = False
            else:
                continue
        filtered_cells.append(cell)
    cells = filtered_cells

    existing_idx = None
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") == "markdown":
            src = "".join(cell.get("source", []))
            if SECTION_TAG in src:
                existing_idx = idx
                break

    if existing_idx is not None:
        cells = cells[:existing_idx]

    notebook["cells"] = cells + cells_to_append
    notebook_path.write_text(json.dumps(notebook, indent=1))


def main() -> None:
    shared_cells = [
        markdown_cell(
            """
            ## Notebook Accuracy Upgrade

            This section keeps the high-accuracy work outside `autoresearch/` while making the notebook experiments repeatable.

            What changed in this section:
            - uses the locally cached BIPIA JSONL dataset instead of requiring a live download
            - compares stronger lexical baselines, including a tuned TF-IDF + Logistic Regression setup
            - adds lightweight structure-aware dense features for tables, line-heavy context, and instruction cues
            - optionally runs a cached local BGE + XGBoost branch when the model is available
            - appends every run to `notebook_results/accuracy_runs.jsonl` so notebook experiments stay auditable
            """
        ),
        code_cell(
            """
            from notebook_utils.accuracy_lab import run_accuracy_suite

            NOTEBOOK_DATASET_LIMIT = 10_000
            NOTEBOOK_CONTEXT_MAX_CHARS = 1_500
            NOTEBOOK_FAST_MODE = True
            NOTEBOOK_INCLUDE_EMBEDDINGS = True

            results_df, run_artifact = run_accuracy_suite(
                dataset_limit=NOTEBOOK_DATASET_LIMIT,
                context_max_chars=NOTEBOOK_CONTEXT_MAX_CHARS,
                include_embeddings=NOTEBOOK_INCLUDE_EMBEDDINGS,
                fast_mode=NOTEBOOK_FAST_MODE,
                experiment_tag="notebook_accuracy_upgrade_v1",
            )

            results_df
            """
        ),
        code_cell(
            """
            print("Experiment log:", run_artifact["log_path"])
            print("Split sizes:", run_artifact["split_sizes"])
            print("Best model summary:")
            run_artifact["best_model"]
            """
        ),
    ]

    append_or_replace_section(ROOT / "TF_IDF.ipynb", shared_cells)
    append_or_replace_section(ROOT / "XG_Boost.ipynb", shared_cells)


if __name__ == "__main__":
    main()
