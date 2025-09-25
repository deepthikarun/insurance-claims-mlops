import os
import json
import joblib
import pandas as pd
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import RootModel
from preprocess import preprocess_raw_df, _build_make_map, make_map


# --------- Config (env or defaults) ----------
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
MAKE_MAP_PATH = os.getenv("MAKE_MAP_PATH", "make_map.json")          # optional
FEATURE_COLUMNS_PATH = os.getenv("FEATURE_COLUMNS_PATH", "feature_columns.json")  # optional
TARGET_COL = "FraudFound_P"  # not expected in inference payloads; dropped if present

# --------- App ----------
app = FastAPI(
    title="Insurance Fraud Predictor",
    description=" FastAPI with prediction.",
    version="1.0",
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapi.json",
)

# ---------- Endpoints (only predictions) ----------
@app.post("/predict-raw")
def predict_raw(rows: List[RawRow]):
    if not rows:
        raise HTTPException(status_code=400, detail="Request body must contain at least one row.")
    assert model is not None and feature_cols is not None

    df_input_raw = pd.DataFrame([r.root for r in rows])
    df_proc = preprocess_raw_df(df_input_raw)

    if TARGET_COL in df_proc.columns:
        df_proc = df_proc.drop(columns=[TARGET_COL])

    # align to training features
    for c in feature_cols:
        if c not in df_proc.columns:
            df_proc[c] = 0
    df_proc = df_proc[feature_cols]

    preds_list = model.predict(df_proc).tolist()
    probs_list = None
    try:
        probs_list = model.predict_proba(df_proc)[:, 1].tolist()
    except Exception:
        pass

    labels = ["Fraud" if p == 1 else "Not Fraud" for p in preds_list]
    return {"predictions": labels, "probabilities_for_Fraud": probs_list}


