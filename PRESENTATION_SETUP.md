# Presentation Setup Guide

A two-part system: a React slide deck (port 5173) + a FastAPI backend (port 8000) that serves the live demo.

---

## Prerequisites

- Python 3.10+ (`python3` must be available)
- `uv` for Python environment/dependency management
- Node.js 18+ with npm

---

## One-Time Setup

### 1. Create the Python virtual environment

Run these commands from the repository root:

```bash
cd /home/kkout/Workspaces/MSC-thesis
uv venv --python python3 .venv
uv pip install --python .venv/bin/python -r fastapi/requirements-api.txt
source .venv/bin/activate
```

Verify that dependencies are installed in the project venv, not in the user/global Python:

```bash
which python
python -c "import sys, numpy, pandas, lightgbm; print(sys.executable); print(numpy.__version__)"
```

### 2. Generate example transactions (requires the venv above)

This creates `fastapi/examples.json` — 3 real transactions used in the live demo.

```bash
cd /home/kkout/Workspaces/MSC-thesis
source .venv/bin/activate
python fastapi/generate_examples.py
```

Expected output:
```
Loading model...
Loading from ieee-fraud-detection-data/processed/test_processed.csv...
Low probability:  prob=0.0xxx
Mid probability:  prob=0.5xxx
High probability: prob=0.9xxx
✅ examples.json written to fastapi/examples.json
```

> Only needs to run once. Re-run only if you delete `examples.json`.

### 3. Install frontend dependencies

```bash
cd /home/kkout/Workspaces/MSC-thesis/presentation
npm install
```

---

## Running the Presentation

Open **two terminals**.

### Terminal 1 — FastAPI backend

```bash
cd /home/kkout/Workspaces/MSC-thesis
source .venv/bin/activate
cd fastapi
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Wait for:
```
All models ready!
INFO: Uvicorn running on http://0.0.0.0:8000
```

> The first startup takes ~10–20 seconds to load LightGBM + cache the SHAP explainer.

### Terminal 2 — React slide deck

```bash
cd /home/kkout/Workspaces/MSC-thesis/presentation
npm run dev
```

Then open: **http://localhost:5173**

---

## Navigation

| Key | Action |
|-----|--------|
| `→` / `Space` | Next slide |
| `←` | Previous slide |
| `G` then number | Jump to slide N |

---

## Slides Overview

| # | Slide | Type |
|---|-------|------|
| 1 | Title | Dark |
| 2 | Roadmap | Dark |
| 3 | The Fraud Detection Challenge | Light |
| 4 | Research Questions | Light |
| 5 | Literature Landscape | Light |
| 6 | Dataset | Light |
| 7 | Methodology Pipeline | Light |
| 8 | Data Split Strategy | Light |
| 9 | EDA Key Findings | Light |
| 10 | Feature Engineering | Light |
| 11 | SHAP-Based Reduction | Light |
| 12 | Experimental Setup | Light |
| 13 | Results — Baseline | Light |
| 14 | Results — Downsampling | Light |
| 15 | Results — Feature Reduction | Light |
| 16 | Results — Threshold Optimization | Light |
| 17 | Champion Scorecard | Dark |
| 18 | Synthesis | Light |
| 19 | Conclusions | Light |
| 20 | **Live Demo** | Dark |
| 21 | Thank You | Dark |

---

## Live Demo (Slide 20)

1. Select a transaction card: **High Risk**, **Low Risk**, or **Borderline**
2. Click **Run Inference →**
3. Watch the 6-stage pipeline animate (Raw Data → Preprocessing → Feature Eng. → Selection → Model → SHAP)
4. Results appear: **Fraud Probability gauge** + **SHAP feature contributions**
5. Click **Try Another Transaction →** to reset

> The demo shows raw fraud probability (0–100%) and which features drove the score up or down.
> No binary FRAUD/LEGIT label and no decision threshold are used in the live demo.

---

## Troubleshooting

**Backend won't start — model not found**
Verify the model artifact exists:
```bash
ls /home/kkout/Workspaces/MSC-thesis/mlruns/161229706116606837/models/m-85ab0e322208453d847c06d654120b9e/artifacts/model.pkl
```

**`ModuleNotFoundError: No module named 'numpy'`**
This means the command is running with a Python environment that does not have the API dependencies installed. A common failure mode is installing NumPy into the user/global Python while the activated `.venv` has `include-system-site-packages = false`, so the venv cannot see it.

Check the interpreter and NumPy location:
```bash
cd /home/kkout/Workspaces/MSC-thesis
source .venv/bin/activate
which python
python -c "import sys, numpy; print(sys.executable); print(numpy.__file__)"
```

If the import fails, install the API dependencies into `.venv`:
```bash
cd /home/kkout/Workspaces/MSC-thesis
uv pip install --python .venv/bin/python -r fastapi/requirements-api.txt
```

**`ValueError: train and valid dataset categorical_feature do not match`**
This means the dataframe passed to LightGBM does not preserve the categorical column metadata used during training. The saved model expects these columns as pandas `category`, not plain object/string columns:

```text
DeviceInfo, P_emaildomain, card6, R_emaildomain, id_31, M4, M5, P_emaildomain_1,
M6, id_30, id_33, ProductCD, R_emaildomain_1, M3, M9, card4
```

The generator and API call `prepare_lightgbm_input(...)` before inference to restore the model feature order and categorical dtypes. If this error returns, verify that inference still goes through that helper before calling `predict_proba` or SHAP.

**API returns `422` with missing feature names**
The live demo sends a partial feature dictionary. This is expected: the backend reindexes the request to the 215 reduced-model features and fills absent columns with `NaN`. The `/predict_explain` endpoint should accept `dict[str, Any]`, not a strict request model with every demo field required.

**`examples.json` examples are not low/mid/high probability**
The generator should select examples by raw probability anchors only: closest to `0.0`, closest to `0.5`, and closest to `1.0`. It should not use the thesis threshold or labels for live-demo example selection.

**Demo shows "Failed to fetch" error**
- Confirm the FastAPI backend is running on port 8000
- Check that `fastapi/examples.json` exists (run step 2 above)

**`examples.json` not found error on `/examples` endpoint**
Run the extraction script (step 2 above).

**Frontend shows blank / crashes**
```bash
cd presentation && npm run build
```
Check for TypeScript errors in the output.

---

## API Endpoints (for reference)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/examples` | Returns the 3 pre-selected example transactions |
| `POST` | `/predict_explain` | Fraud probability + top-15 SHAP values |
| `POST` | `/predict` | Binary prediction (legacy) |

All endpoints require header: `X-API-Key: c1c58f5a-8f7c-4bdb-9d78-1c3b12c9f3f2`

---

## Model Details

- **Model**: LightGBM (`LGBM_Optuna_Reduced`)
- **ROC-AUC**: 0.9191 | **PR-AUC**: 0.5737
- **Live demo output**: raw fraud probability only; example cards are selected near 0%, 50%, and 100%
- **Input**: 81 engineered features (reindexed to 215 training features; missing → NaN)
- **Artifacts**: `mlruns/161229706116606837/models/m-85ab0e322208453d847c06d654120b9e/artifacts/`
