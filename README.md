# Prompt Injection Detection 

Dataset used in notebook:
- `MAlmasabi/Indirect-Prompt-Injection-BIPIA-GPT`

## What Was Implemented

### 1) Baseline ONNX classifier (DeBERTa prompt injection model)

Implemented:
- Loaded `protectai/deberta-v3-base-prompt-injection-v2` using `transformers` + `optimum.onnxruntime`
- Built text as:
  - `Context: ...`
  - `User intent: ...`
- Evaluated on `train[:500]`

Observed result:
- Accuracy: ~`50.2%`
- F1: very low for attack class (`~0.14` range in baseline cell output)

Takeaway:
- Baseline model underperformed on this setup/data formatting split.

---

### 2) Threshold sweep on baseline probabilities

Implemented:
- Swept thresholds from `0.1` to `0.9`

Observed result:
- Only marginal change; performance stayed around baseline quality.

Takeaway:
- Threshold tuning alone cannot fix a weak base signal.

---

### 3) Embedding + XGBoost branch

Implemented:
- Sentence embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- Trained `XGBClassifier` on 2k samples

Observed result:
- `XGBoost Results: Accuracy 0.7550, F1 0.7667`

Then tried BGE embeddings:
- `BAAI/bge-small-en-v1.5`
- Result improved slightly:
  - `Accuracy 0.7600, F1 0.7703`

Takeaway:
- Embedding + tree model gave a major jump over baseline and is practical.

---

### 4) Scale-up to 10k + ensemble

Implemented:
- Scaled training data to 10k
- Trained stronger XGBoost config
- Tried soft-voting ensemble (`XGBoost + RandomForest`)

Observed result:
- XGBoost on 10k: `Accuracy 0.8305, F1 0.8328`
- Ensemble on 10k: `Accuracy 0.8290, F1 0.8310`
- Summary cell output:
  - DeBERTa baseline (500): `50.2%`
  - XGBoost (2k): `75.5%`
  - XGBoost/10k summary line: `82.9%`

Takeaway:
- Bigger data + embedding branch was the main gain driver.
- Ensemble did not materially beat tuned XGBoost.

---

### 5) Added a clean upgrade pipeline

Implemented new section:
- `  Accuracy Upgrade Pipeline (TF-IDF + Embeddings + Fusion)`

Contains:
- Proper split (`train/val/test`)
- Branch A: `TF-IDF (word + char n-grams) + LogisticRegression`
- Branch B: `Embeddings + XGBoost`
- Fusion: meta Logistic Regression over branch probabilities
- Validation-based threshold calibration

Observed output in notebook:
- Fusion rows around `~0.91` test metrics in displayed table
- Calibrated thresholds printed (example output includes XGBoost threshold `0.370`)
- Best-model line currently prints:
  - `Best model on test by F1: TF-IDF + LogReg (threshold=0.520)`

Takeaway:
- Combining lexical + semantic signals significantly improved results versus earlier single-branch baselines.


### Practical scope issue

##Implementation:
  - Threshold tuning
  - Embedding/XGBoost path
  - 10k scale-up + ensemble
  - New TF-IDF + Embedding + Fusion pipeline


## What I Understood

- A pre-trained classifier can fail badly if data style/domain differs(DeBERTa was for direct prompt injection, And datasets can in indirect).
- Threshold tuning helps only when model signal is already meaningful.
- Lexical models (TF-IDF + linear) are excellent at explicit attack patterns.
- Embedding models capture paraphrased/semantic attacks better.
- Fusion works because lexical and semantic branches fail on different samples.
- Proper validation split + threshold calibration is important for stable, realistic performance.
