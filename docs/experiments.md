# Research Narrative And Experiment History

## Objective

The repository investigates binary classification for indirect prompt injection detection. The central question is not only whether a model can separate safe from malicious examples, but which representation strategy remains robust when the attack signal is embedded inside ordinary-looking context.

The workflow moved through seven stages:

1. Imported-model baseline.
2. Embedding-based classical ML baseline.
3. Scale-up and ensemble testing.
4. Reproducible fusion pipeline with threshold calibration.
5. Package extraction into `autoresearch/`.
6. Stricter package protocol upgrades.
7. Notebook-only accuracy upgrade outside `autoresearch/`.

## Stage 1: Pretrained DeBERTa Baseline

Primary source notebooks:

- [`TF_IDF.ipynb`](../TF_IDF.ipynb)
- [`XG_Boost.ipynb`](../XG_Boost.ipynb)

### What was implemented

- `protectai/deberta-v3-base-prompt-injection-v2`
- Inference through `transformers` and `optimum.onnxruntime`
- Input formatting as:
  - `Context: ...`
  - `User intent: ...`
- Initial evaluation on `train[:500]`

### Why this approach was reasonable

Using a pretrained prompt-injection model is the fastest way to test whether the dataset is already aligned with an available detector. It establishes a useful "buy versus build" baseline before investing in custom feature engineering.

### Reported result

Observed notebook outputs show:

- Accuracy around `0.500-0.502`
- Attack-class F1 around `0.138-0.144`
- Heavy bias toward predicting the safe class

### Interpretation

This baseline failed for the intended use case. The likely reason is task mismatch:

- The imported model appears stronger for direct or instruction-style prompt injection.
- The BIPIA-style indirect setup often hides the attack signal inside retrieved context or mixed content.

The result is important because it justifies moving away from pure transfer learning without adaptation.

## Stage 2: Threshold Sweeping

### What was implemented

- Probability extraction from the DeBERTa classifier
- Threshold sweep from `0.1` to `0.9`

### Why this approach was tried

Threshold tuning is the least invasive intervention when a model is miscalibrated but still rank-orders examples well.

### Reported result

Across both notebooks, threshold tuning produced only marginal variation:

- Accuracy remained near `0.50`
- Attack-class F1 stayed very low

### Interpretation

The weak performance was not mainly a threshold problem. The base ranking signal itself was poor. This is a useful negative result because it prevents overfitting the evaluation procedure around a fundamentally weak detector.

## Stage 3: Embeddings Plus XGBoost

### What was implemented

- Sentence embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- `XGBClassifier` on embedded text
- Training on roughly `2k` examples
- Later replacement of MiniLM embeddings with `BAAI/bge-small-en-v1.5`

### Why this approach was stronger

This design separates representation learning from the final classifier:

- Sentence embeddings capture semantic similarity and paraphrase patterns.
- XGBoost can learn non-linear decision boundaries over those dense features.
- The training cost is much lower than fine-tuning a transformer end-to-end.

### Reported result

Notebook outputs record:

- MiniLM + XGBoost: `accuracy=0.7550`, `f1=0.7667`
- BGE + XGBoost: `accuracy=0.7600`, `f1=0.7703`

### Interpretation

This is the first clear jump in quality. It indicates that the problem benefits from semantic representation, but the gain is still incomplete because the model does not fully exploit lexical artifacts such as token-level attack patterns, template fragments, or table-specific anomalies.

## Stage 4: Larger Dataset And Ensemble Testing

### What was implemented

- Increased dataset usage to `10k` examples
- Stronger XGBoost configuration
- Soft-voting ensemble with `XGBoost + RandomForest`

### Why this approach was tried

Once a representation is promising, two obvious next questions are:

1. Does more data continue to help?
2. Does a second tree model add complementary signal?

### Reported result

Notebook outputs show:

- XGBoost on `10k`: `accuracy=0.8305`, `f1=0.8328`
- Ensemble on `10k`: `accuracy=0.8290`, `f1=0.8310`

### Interpretation

More data helped materially. The ensemble did not. That suggests the main bottleneck was representation plus sample coverage, not insufficient model diversity inside tree-based learners.

## Stage 5: Lexical-Semantic Fusion

Primary source notebook:

- [`TF_IDF.ipynb`](../TF_IDF.ipynb)

### What was implemented

- Proper `train/val/test` split
- Branch A: word and character TF-IDF with Logistic Regression
- Branch B: embeddings with XGBoost
- Meta-model fusion over branch probabilities
- Validation-based threshold calibration

### Why this approach was chosen

This is the most principled design in the repository because the branches target different failure modes:

- TF-IDF is strong on literal phrases, unusual n-grams, and local syntax patterns.
- Embeddings are stronger on paraphrased or semantically indirect attacks.
- Fusion combines the branch outputs instead of forcing one representation to solve every regime.

### Reported result

The notebook shows:

- Calibrated thresholds around:
  - TF-IDF: `0.520`
  - XGBoost: `0.370`
  - Fusion: `0.460`
- Test metrics in the displayed table around:
  - TF-IDF + Logistic Regression: roughly `0.912` accuracy and `0.9135` F1
  - Fusion: roughly `0.910` accuracy and `0.9103` F1 at default threshold `0.5`

### Interpretation

The threshold calibration result matters more than the raw `0.5` threshold comparison. The repository's later package formalizes that insight: the final operating point should be selected on validation data, not assumed.

## Stage 6: AutoResearch Package

Primary source code:

- [`autoresearch/`](../autoresearch)
- [`AutoResearch.ipynb`](../AutoResearch.ipynb)

### What was implemented

- Deterministic split logic
- Configured search space over TF-IDF, embeddings, XGBoost, and fusion parameters
- Validation-first model selection
- FPR-constrained threshold search
- Result logging to JSONL
- Error mining export for hard false positives and false negatives

### Why this is the final form

The notebook exploration proved the modeling idea. The package turns it into a repeatable procedure:

- same seed
- same split
- same comparison rules
- same logging structure

This is the correct transition from experimentation to reproducible research engineering.

### Best recorded result

The best run in `autoresearch_results/runs.jsonl` is `s4`:

- Validation: `accuracy=0.922667`, `f1=0.923885`
- Test: `accuracy=0.924051`, `f1=0.924303`
- Embedding model: `BAAI/bge-small-en-v1.5`
- TF-IDF n-grams: word `(1, 3)`, char `(3, 6)`
- Fusion `C=2.0`
- Threshold: `0.4604`

Important qualification:

These saved artifacts reflect the earlier package version, where the fusion layer was trained on validation predictions. The current codebase has been upgraded to train fusion on out-of-fold training predictions and to use exact score-based threshold search. That means the next rerun should be interpreted as the new baseline for fair comparison.

## Stage 7: Notebook-Only Accuracy Upgrade

Primary source files:

- [`TF_IDF.ipynb`](../TF_IDF.ipynb)
- [`XG_Boost.ipynb`](../XG_Boost.ipynb)
- [`notebook_utils/accuracy_lab.py`](../notebook_utils/accuracy_lab.py)
- [`notebook_results/accuracy_runs.jsonl`](../notebook_results/accuracy_runs.jsonl)

### What was implemented

- a cache-first local dataset loader that reads the BIPIA JSONL snapshot directly
- deterministic `train/val/test` splitting outside `autoresearch/`
- a tuned lexical baseline with word `(1, 3)` and char `(3, 6)` TF-IDF n-grams
- a denser lexical variant that adds lightweight structural signals:
  - context length
  - newline density
  - table / pipe density
  - code-fence count
  - instruction-pattern hits
- validation-selected thresholding for every notebook-side run
- JSONL experiment logging for notebook executions

### Why this approach was added

There was a user constraint to avoid changing `autoresearch/` while still improving accuracy. The notebooks already contained the right modeling direction, but the implementation was fragmented and hard to rerun cleanly. This upgrade moved the accuracy-oriented notebook logic into a small helper module and made the notebook path reproducible enough to compare runs.

### Recorded result

From `notebook_results/accuracy_runs.jsonl`, the strongest full notebook run under `experiment_tag="notebook_accuracy_upgrade_v1"` is:

- model: `tfidf_logreg_tuned_plus_dense`
- validation F1: `0.935013`
- test accuracy: `0.929333`
- test F1: `0.930079`
- test ROC-AUC: `0.979398`
- test PR-AUC: `0.976407`
- threshold: `0.5000`

The same log also shows:

- `tfidf_logreg_baseline`: `accuracy=0.911333`, `f1=0.912211`
- `tfidf_logreg_tuned`: `accuracy=0.918667`, `f1=0.919842`
- `bge_small_xgb_tuned`: `accuracy=0.863333`, `f1=0.870006`

### Interpretation

This notebook result is important for two reasons:

1. A tuned lexical model with a few structure-aware dense signals outperformed the notebook-side embedding branch by a wide margin.
2. The notebook-only path slightly exceeds the older saved `s4` test metrics, although the protocols are still not identical enough to claim a strict apples-to-apples win over the package result.

The practical conclusion is that the next accuracy iteration should likely keep lexical features central and treat semantic features as optional augmentation rather than the default driver.

## Main Research Conclusions

1. Off-the-shelf prompt injection detectors do not automatically transfer to indirect prompt injection datasets.
2. Threshold tuning cannot rescue a model whose ranking signal is weak.
3. Semantic embeddings plus classical ML provide a strong practical baseline.
4. More data improves the embedding-based branch substantially.
5. Lexical and semantic branches capture different evidence, so fusion is justified.
6. Validation-based threshold selection is part of the model, not merely a reporting detail.
7. In the current notebook-only path, tuned lexical features plus simple structure-aware signals outperform the standalone embedding branch.
