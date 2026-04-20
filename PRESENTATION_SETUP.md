# Presentation Setup Guide

A two-part system: a React slide deck (port 5173) + a FastAPI backend (port 8000) that serves the live demo.

---

## Prerequisites

- Python 3.13+ (via pyenv — already installed)
- Node.js 18+ with npm

---

## One-Time Setup

### 1. Create the Python virtual environment

```bash
python3 -m venv /mnt/d/Developer/MSC-thesis/.venv
source /mnt/d/Developer/MSC-thesis/.venv/bin/activate
pip install --upgrade pip
pip install -r /mnt/d/Developer/MSC-thesis/fastapi/requirements-api.txt
```

### 2. Generate example transactions (requires the venv above)

This creates `fastapi/examples.json` — 3 real transactions used in the live demo.

```bash
cd /mnt/d/Developer/MSC-thesis
source .venv/bin/activate
python fastapi/generate_examples.py
```

Expected output:
```
Loading model...
Loading from ieee-fraud-detection-data/processed/train_processed.csv...
Clear fraud:  prob=0.9xxx
Clear legit:  prob=0.0xxx
Borderline:   prob=0.04xx
✅ examples.json written to fastapi/examples.json
```

> Only needs to run once. Re-run only if you delete `examples.json`.

### 3. Install frontend dependencies

```bash
cd /mnt/d/Developer/MSC-thesis/presentation
npm install
```

---

## Running the Presentation

Open **two terminals**.

### Terminal 1 — FastAPI backend

```bash
cd /mnt/d/Developer/MSC-thesis
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
cd /mnt/d/Developer/MSC-thesis/presentation
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
> No binary FRAUD/LEGIT label — threshold selection is a key thesis contribution.

---

## Troubleshooting

**Backend won't start — model not found**
Verify the model artifact exists:
```bash
ls /mnt/d/Developer/MSC-thesis/mlruns/161229706116606837/models/m-85ab0e322208453d847c06d654120b9e/artifacts/model.pkl
```

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
- **Cost-optimal threshold**: 0.0419 → Recall 92.7%, Precision 8.9%
- **Input**: 81 engineered features (reindexed to 215 training features; missing → NaN)
- **Artifacts**: `mlruns/161229706116606837/models/m-85ab0e322208453d847c06d654120b9e/artifacts/`
