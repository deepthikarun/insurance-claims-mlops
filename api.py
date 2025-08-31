# api.py — Serve-only FastAPI (loads model.pkl; no training)
from fastapi import FastAPI, HTTPException
from pydantic import RootModel
from typing import List, Dict, Any, Optional
import os, joblib, pandas as pd

from preprocess import preprocess_raw_df  # shared
# Keep these for consistent filenames/columns
CSV_PATH = os.getenv("CSV_PATH", "fraud_oracle.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
TARGET_COL = os.getenv("TARGET_COL", "FraudFound_P")

app = FastAPI(
    title="Insurance Fraud Model API",
    description="Loads a pre-trained model.pkl and serves predictions.",
    version="4.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

model = None
feature_cols: Optional[list] = None

def load_model_and_schema():
    """Load model.pkl and infer feature columns from CSV using preprocess (without target)."""
    global model, feature_cols
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}. Deploy a build that includes model.pkl.")

    model_obj = joblib.load(MODEL_PATH)
    if not hasattr(model_obj, "predict"):
        raise RuntimeError("Loaded object is not a valid model.")
    model = model_obj

    # Infer feature schema from CSV (or optionally store alongside the model)
    if not os.path.exists(CSV_PATH):
        # If CSV isn't shipped to prod, you could store feature_cols in a separate json file during training.
        raise RuntimeError(f"CSV not found at {CSV_PATH} to infer feature columns. "
                           f"Consider exporting feature schema during training.")
    df_raw = pd.read_csv(CSV_PATH)
    df_prep = preprocess_raw_df(df_raw)
    X = df_prep.drop(columns=[TARGET_COL]) if TARGET_COL in df_prep.columns else df_prep
    feature_cols = list(X.columns)

@app.on_event("startup")
def _startup():
    load_model_and_schema()

class RawRow(RootModel[Dict[str, Any]]): 
    pass

class NumericRow(RootModel[Dict[str, Any]]): 
    pass

@app.get("/")
def root():
    return {"status": "ok", "message": "Use /predict-raw (strings) or /predict (numeric)."}

@app.get("/health")
def health():
    if model is None or feature_cols is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {"status": "healthy", "num_features": len(feature_cols)}

@app.post("/predict-raw")
def predict_raw(rows: List[RawRow]):
    if model is None or feature_cols is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    df_input_raw = pd.DataFrame([r.root for r in rows])
    df_proc = preprocess_raw_df(df_input_raw)
    if TARGET_COL in df_proc.columns:
        df_proc = df_proc.drop(columns=[TARGET_COL])
    for c in feature_cols:
        if c not in df_proc.columns:
            df_proc[c] = 0
    df_proc = df_proc[feature_cols]
    preds = model.predict(df_proc)
    try:
        probs = model.predict_proba(df_proc)[:, 1].tolist()
    except Exception:
        probs = None
    return {"predictions": preds.tolist(), "probabilities_for_class_1": probs}

@app.post("/predict")
def predict_numeric(rows: List[NumericRow]):
    if model is None or feature_cols is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    df = pd.DataFrame([r.root for r in rows])
    for c in list(df.columns):
        if str(c).lower().startswith("unnamed:"):
            df = df.drop(columns=[c])
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0
    df = df[feature_cols]
    preds = model.predict(df)
    try:
        probs = model.predict_proba(df)[:, 1].tolist()
    except Exception:
        probs = None
    return {"predictions": preds.tolist(), "probabilities_for_class_1": probs}
