from fastapi import FastAPI, Depends
import mlflow.pyfunc
from config import API_KEY, API_KEY_NAME
from model import PredictionRequest
from fastapi.security.api_key import APIKeyHeader
from fastapi import Security, HTTPException

import xgboost

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def authenticate(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(status_code=403, detail="Invalid API Key")

import mlflow.pyfunc

MODEL_PATH = "/home/kkout/Workspaces/MSC-thesis/mlruns/266162046764656308/models/m-9daf2e7529b64c69aa55be1aa8ed5bd6/artifacts"

print("Loading MLflow model...")
ml_model = mlflow.pyfunc.load_model(MODEL_PATH)
print("Model Loaded!")

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="ML Model Prediction API",
    description="A FastAPI service for feeding input into an MLflow model.",
    version="1.0.0"
)


# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict(data: PredictionRequest, api_key: str = Depends(authenticate)):
    # Convert input → MLflow model input format
    input_df = data.model_dump()
    prediction = ml_model.predict([input_df])
    return {"prediction": prediction[0]}