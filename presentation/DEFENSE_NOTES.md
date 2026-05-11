# Thesis Presentation Defense Notes

## Core Positioning

This presentation defends a controlled experimental study on machine learning for fraud detection using the IEEE-CIS dataset. The thesis does not claim a production-ready banking system. It claims that gradient boosting models, behavioral feature engineering, majority-class downsampling, SHAP-based feature reduction, and threshold tuning can be evaluated under a defensible chronological protocol.

The strongest one-sentence framing:

> The study prioritizes leakage-safe model comparison first, then interprets how model choice, feature design, sampling, and threshold selection affect practical fraud-detection behavior.

## Defensibility Points

- The data split is chronological: first 80% for training, final 20% as held-out test data.
- The test set was not used for Optuna tuning, feature selection decisions, or threshold selection.
- The test set was not downsampled; downsampling was applied only to the training data.
- Time-series cross-validation was used inside the training period to reduce temporal leakage risk.
- ROC-AUC is used as the primary benchmark metric because it is threshold-independent and aligns with the Kaggle competition protocol.
- PR-AUC, precision, recall, and F1 are also reported because ROC-AUC alone can be optimistic under class imbalance.
- Accuracy is intentionally not emphasized because a 96.5% non-fraud majority makes accuracy misleading.
- The live demo is illustrative: it demonstrates the pipeline idea, probability output, and local SHAP explanation, not a validated production service.
- Behavioral aggregate features use simulated user anchors from anonymized fields, not real account IDs.
- The thesis uses public benchmark data, so conclusions are about reproducible experimental behavior, not direct deployment in a specific financial institution.

## LightGBM Defense

If asked why LightGBM performed best:

- The thesis attributes LightGBM's strong performance partly to histogram-based split finding.
- This is suitable for the high-dimensional IEEE-CIS feature space because continuous features are discretized into bins before candidate splits are evaluated.
- The binning process can reduce computational cost and may act as mild regularization under severe class imbalance.
- Empirically, LightGBM had the most stable ROC-AUC range across configurations: approximately 0.917 to 0.919.
- LightGBM also achieved the strongest baseline balance at the default threshold: highest ROC-AUC, PR-AUC, and F1.

Safe wording:

> I would not claim LightGBM is universally best for fraud detection. In this dataset and protocol, it was the most stable ROC-AUC performer, and the thesis gives a plausible explanation based on histogram-based split finding in a high-dimensional feature space.

## Scope Boundaries

- Do not claim the model is production-ready.
- Do not claim the threshold of 0.1 is optimal.
- Do not claim the simulated user ID is equivalent to a real customer ID.
- Do not claim SHAP proves causality; it explains model attribution, not causal fraud drivers.
- Do not claim Kaggle leaderboard performance was the goal; the thesis uses a stricter chronological holdout.
- Do not claim downsampling always improves all metrics; it improved some behaviors while changing precision-recall trade-offs.

## Likely Committee Questions

### Why not use accuracy?

Because only about 3.5% of transactions are fraudulent. A classifier can achieve high accuracy by predicting almost everything as legitimate, while failing the actual fraud-detection objective. ROC-AUC, PR-AUC, precision, recall, and F1 are more informative.

### Why is ROC-AUC primary if PR-AUC matters more under imbalance?

ROC-AUC is useful for threshold-independent ranking comparison and aligns with the IEEE-CIS/Kaggle benchmark. PR-AUC is reported alongside it because it better reflects minority-class detection quality under severe imbalance.

### Why chronological split instead of random split?

Fraud detection is temporal. Random splitting can leak future patterns into training and inflate performance. Chronological splitting better simulates training on historical transactions and predicting future ones.

### Why downsampling instead of SMOTE?

The thesis tested oversampling but treated SMOTE as less suitable because synthetic fraud transactions may introduce artificial behavior patterns. Downsampling is simpler and avoids generating synthetic fraud examples.

### Did downsampling make the test easier?

No. The held-out test set remained in its original imbalanced distribution. Only the training data was downsampled.

### Why did threshold tuning reduce precision so much?

Lowering the threshold increases the number of flagged transactions. This captures more fraud, increasing recall, but also flags many more legitimate transactions, reducing precision. This is the operational trade-off.

### Is threshold 0.1 the recommended production threshold?

No. It is an experiment showing the effect of a lower operating point. A production threshold should be chosen using review capacity, false-positive cost, false-negative cost, and calibration.

### Does SHAP prove these features cause fraud?

No. SHAP explains how the trained model uses features to make predictions. It supports interpretability and feature selection, but it does not establish causality.

### Why are Kaggle winning scores higher?

The thesis uses a conservative chronological holdout and avoids leaderboard-driven optimization. Winning Kaggle solutions often use extensive ensembling, blending, and feedback from public leaderboard iteration.

## Presentation Strategy Notes

- Emphasize that the thesis is careful about leakage and operational interpretation.
- When challenged, separate model ranking from deployment decisioning.
- Use the phrase "under this dataset and protocol" when discussing LightGBM superiority.
- For threshold tuning, say "operational lever" rather than "model improvement."
- For feature engineering, say "behavioral proxies from anonymized data" rather than "customer behavior" if precision is needed.
- For SHAP, say "model explanation" rather than "reason for fraud."
