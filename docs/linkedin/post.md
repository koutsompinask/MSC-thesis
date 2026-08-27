# LinkedIn post

## Post

About 3.5% of transactions in a real payments dataset are fraudulent. A model that approves everything scores 96.5% accuracy and is worthless.

That was the starting point of my MSc thesis at the University of Athens, on 590K+ labelled card transactions.

I approached it as a system, not just a notebook experiment:
• Chronological splits to avoid future information leaking into training
• Optuna tuning inside the training window, never touching the holdout
• MLflow tracking for experiments and models
• Bootstrap confidence intervals, to tell real gaps from noise
• SHAP-based feature analysis, reducing 748 features to 215
• A FastAPI service that returns both a fraud probability and the factors behind it

The threshold mattered more than the model. Moving it from 0.5 to 0.1 took recall from 0.64 to 0.86 and precision from 0.40 to 0.14. The model ranks risk. The business decides where to draw the line.

There is still work to do: calibration, drift monitoring, and a cost model that can turn threshold selection into an explicit business decision.

But that's also been one of the biggest lessons from this project:

Building the model is only part of building the system.

Grade: 10/10.
Results, code, full write up: github.com/koutsompinask/MSC-thesis

#MachineLearning #FraudDetection #MLOps #SoftwareEngineering

## Media

Document post: `Machine Learning for Fraud Detection.pdf` (12 slides, 2.2 MB).
LinkedIn uses the filename as the document title shown above the carousel, so keep the name as is.

A document post cannot also carry images or video, so the three screenshots are not attached
separately. The carousel already contains the SHAP and live demo visuals.

## Before posting

- [ ] Commit and push. The README is the payoff for the link and is not on GitHub yet.
- [ ] Open github.com/koutsompinask/MSC-thesis and check the README renders, images included.
- [ ] Upload the PDF as a document post, confirm the title reads "Machine Learning for Fraud Detection".
- [ ] Swipe all 12 slides on a phone before publishing.
- [ ] Tag your supervisor and the university if you want the academic reach.
- [ ] Tuesday to Thursday, 09:00 to 11:00 EET.
