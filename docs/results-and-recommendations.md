# Results, Limitations, And Recommended Corrections

## Best Recorded Outcome

From `autoresearch_results/runs.jsonl`, the best saved run is `s4`:

- validation accuracy: `0.922667`
- validation F1: `0.923885`
- test accuracy: `0.924051`
- test F1: `0.924303`
- test ROC-AUC: `0.976352`
- test PR-AUC: `0.973685`
- embedding model: `BAAI/bge-small-en-v1.5`
- threshold: `0.4604`

This is a strong result for the repository's current design and a substantial improvement over the imported-model baseline.

Important qualification:

These metrics come from the saved historical artifacts. The code now uses a stricter protocol with out-of-fold fusion training and exact threshold selection, so the next rerun may report slightly lower but more trustworthy validation and test numbers.

The repository now also includes a layered defense pass. That means you should compare two kinds of performance after rerunning:

- base detector metrics
- routed metrics
- sanitized-context metrics and defense gate behavior

## Best Current Notebook-Only Outcome

The latest full notebook run recorded in `notebook_results/accuracy_runs.jsonl` is currently stronger than the older notebook baselines:

- model: `tfidf_logreg_tuned_plus_dense`
- validation F1: `0.935013`
- test accuracy: `0.929333`
- test F1: `0.930079`
- test ROC-AUC: `0.979398`
- test PR-AUC: `0.976407`

This run comes from the notebook-only helper path rather than `autoresearch/`, so it should be treated as the current notebook reference, not as a replacement for the package benchmark.

## What Worked Best

### 1. The lexical branch is stronger than expected

The saved runs show that the TF-IDF branch alone is already competitive. In the best configuration, validation TF-IDF F1 is above `0.918`, which means local lexical evidence is a major source of signal in this dataset.

The latest notebook-only result strengthens that conclusion further. The tuned lexical-plus-dense variant reached test F1 above `0.930`, which is better than the standalone notebook embedding branch and better than the older notebook lexical baselines.

### 2. The embedding branch is useful but not dominant

The embedding branch improves semantic coverage, but on its own it underperforms the lexical branch. Its value is mainly complementary.

The latest notebook log reinforces this: `bge_small_xgb_tuned` reached `accuracy=0.863333`, `f1=0.870006`, well below the tuned lexical variants.

### 3. Fusion improves the final operating point

Fusion outperforms either branch alone in the recorded package runs because it combines a high-precision lexical detector with a semantically broader secondary model.

### 4. Threshold calibration is not optional

The best threshold is below `0.5`. That confirms the repository's later design choice to calibrate the operating point on validation data instead of treating `0.5` as fixed.

## What The Hard Errors Suggest

The exported `errors_s4.json` file shows several recurring patterns among hard false negatives:

- questions over tables and structured records
- benign-looking factual queries whose attack label depends on hidden contextual behavior
- examples where the context is long, noisy, or mixed-format
- cases where the user intent looks ordinary in isolation

This implies that remaining errors are not only about obvious malicious phrasing. The unresolved challenge is often contextual interaction between content and intent.

## Recommended Corrections

## 1. Keep one notebook path and treat the helper module as the source of truth

Current issue:

- `TF_IDF.ipynb` and `XG_Boost.ipynb` still overlap heavily.
- historical exploratory cells remain noisier than the new helper-backed section.

Correction:

- keep one canonical notebook-side accuracy section
- keep `notebook_utils/accuracy_lab.py` as the implementation source of truth
- keep `AutoResearch.ipynb` separate as the package orchestration notebook

Why this is better:

It reduces drift between experimental records and prevents notebook JSON from becoming the only place where model logic lives.

## 2. Tighten the embedding cache key

Current issue:

- the cache key is heuristic rather than content-complete

Correction:

- hash all texts or hash a manifest derived from the exact split contents

Why this is better:

It makes cache reuse safer when multiple datasets or alternative formatting choices are introduced later.

## 3. Improve configuration-to-result traceability

Current issue:

- results are logged, but package versions, notebook provenance, and dataset revision information are not recorded

Correction:

- store package versions, timestamp, dataset revision if available, and code commit hash alongside each run

Why this is better:

It raises the work from good local experimentation to stronger research reproducibility.

## 4. Compare notebook lexical-plus-dense against package fusion under one protocol

Current issue:

- the current notebook winner and the older package winner were produced under different execution paths

Correction:

- compare:
  - notebook `tfidf_logreg_tuned_plus_dense`
  - package TF-IDF only
  - package fusion always
  - package routed TF-IDF/fusion

Why this is better:

It separates real modeling improvement from protocol differences and clarifies whether the extra semantic branch is still worth its compute cost.

## 5. Validate the layered defense against adaptive cases

Current issue:

- the new defense layer is heuristic and has not yet been benchmarked against adaptive or obfuscated attacks

Correction:

- add targeted evaluation for:
  - instruction paraphrases
  - HTML/Markdown hiding
  - tool-call phrasing
  - table-cell injections

Why this is better:

It tests whether chunk removal and alignment gating are robust enough to matter outside clean benchmark examples.

## 6. Deepen slice-based evaluation

Current issue:

- the package now reports a first set of heuristic slices, but the slice taxonomy is still simple

Correction:

- add richer slices such as:
  - retrieval source type
  - context truncation hit vs no truncation
  - explicit attack phrase vs implicit attack
  - domain bucket if multiple corpora are added later

Why this is better:

It makes the new slice-metric framework more diagnostic and more useful for targeted data collection.

## 7. Consider stronger semantic branches only after protocol cleanup

Current issue:

- it is tempting to add larger embedding models or fine-tuned transformers immediately

Correction:

- first improve the evaluation protocol, cache identity, and slice analysis
- then test larger encoders or supervised fine-tuning

Why this is better:

A stronger model is only a real improvement if the surrounding experiment design is already trustworthy.

## What Feels Like The Right Next Research Step

The package now implements the first protocol cleanup steps. The next best step is to build on that stricter foundation:

1. rerun the sweep under the new protocol
2. compare old and new slice metrics
3. review hard false negatives under the updated operating point
4. then expand targeted data collection

That path is more likely to produce trustworthy gains than simply increasing model size.

## Final Assessment

The repository already contains a credible research direction:

- it starts with a negative baseline result
- it demonstrates a meaningful representation shift
- it shows scale effects
- it justifies fusion through complementary branches
- it preserves final runs in a reproducible package

What is missing is not the core idea. What remains is methodological tightening and better experimental reporting discipline.
