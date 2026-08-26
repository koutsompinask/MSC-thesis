# LinkedIn post — MSc thesis (10/10)

## Recommended post

My MSc thesis on fraud detection was graded 10/10.

The part I'm proudest of isn't the model score. It's that anyone can clone the repo and reproduce every number in it.

Fraud is rare: about 3.5% of transactions in the IEEE-CIS dataset. A model that calls everything legitimate is 96.5% accurate and completely useless. The real question was never accuracy. It's how much fraud you accept missing so you don't block paying customers.

I built it like a system, not a notebook.

Chronological split, not random. Trained on the first 80% by transaction time, kept the last 20% untouched. A random split scores higher and lies to you, because fraud patterns move over time.

Optuna tuning run entirely inside the training window, with expanding time-series cross-validation. The holdout never informed tuning, feature selection or the threshold.

MLflow on every run: params, metrics, artifacts. Months later, "which config produced this number?" was still answerable. Including at the defense.

Every metric logged with bootstrap confidence intervals, 200 resamples each. That is what tells you whether a gap between two models is a real difference or noise.

SHAP for cross-model agreement. 748 engineered features down to 215, keeping only what 2 of 3 models ranked in their top 30%. Performance held.

A FastAPI service loading the logged model, returning a fraud probability plus a per-transaction explanation. A model that can't answer "why did you flag this?" doesn't survive contact with a fraud-ops team.

What stayed with me isn't which model won. On the final model, moving the decision threshold from 0.5 to 0.1 took recall from 0.64 to 0.86 and precision from 0.40 to 0.14. One number, chosen by a human, swinging the system harder than any gap between LightGBM, XGBoost and CatBoost. The model ranks risk. The business decides where to cut, based on review capacity and what a false alarm costs.

Scope, honestly: public benchmark data, simulated user anchors, an illustrative demo. Not a production fraud system.

Thanks to my supervisor and committee. After three years building enterprise backends, the lesson that stuck is how much of ML is ordinary engineering: reproducibility, traceability, and being honest about what your evaluation actually proves.

Code and notebooks in the first comment.

#MachineLearning #FraudDetection #MLOps #DataScience #SoftwareEngineering

---

## Alternative hook (engineering-first)

Replace the first two lines with:

> Any model can hit 0.92 ROC-AUC on a random split. That's exactly why I didn't use one.

...and move the grade to the closing block:

> Thanks to my supervisor and committee for the 10/10. After three years building enterprise backends, the lesson that stuck is how much of ML is ordinary engineering: reproducibility, traceability, and being honest about what your evaluation actually proves.

---

## First comment (post immediately after publishing)

Repo, notebooks and the FastAPI demo service: https://github.com/koutsompinask/MSC-thesis

Dataset: IEEE-CIS Fraud Detection (Kaggle), 590,540 labeled transactions.
Stack: Python, LightGBM / XGBoost / CatBoost, Optuna, MLflow, SHAP, FastAPI.

---

## Media (attach in this order)

1. `demo.png` — the live demo: a scored transaction with its probability gauge and SHAP waterfall.
2. `mlflow-runs.png` — the MLflow experiment table, evidence of the tracked pipeline.
3. `shap-summary.png` — global feature attributions after reduction.

Export at 1200x1200 or 1200x627. Check the SHAP labels are readable at ~390px wide.

## Publishing checklist

- [ ] Link in the FIRST COMMENT, not the body.
- [ ] Tag supervisor / NKUA if you want the academic reach.
- [ ] Post Tue-Thu, 09:00-11:00 EET.
- [ ] Don't add the 9.58 GPA. The 10/10 is enough.
