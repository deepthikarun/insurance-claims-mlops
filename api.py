import os
import json
import joblib
import pandas as pd
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import RootModel

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

# ---------- Encoders / preprocessing ----------
day_map = {"Sunday":0,"Monday":1,"Tuesday":2,"Wednesday":3,"Thursday":4,"Friday":5,"Saturday":6}
month_map = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
bool_map = {"No":0, "Yes":1}
agent_map = {"External":0, "Internal":1}
veh_map = {"Sport":0, "Sedan":1, "Utility":2}
marital_map = {"Widow":0, "Single":1, "Married":2, "Divorced":3}
sup_map = {"none":0, "1 to 2":1.5, "3 to 5":4, "more than 5":6}
base_policy_map = {"Liability":0, "Collision":1, "All Perils":2}
addr_change_map = {"1 year":1, "no change":0, "4 to 8 years":6, "2 to 3 years":2.5, "under 6 months":0.3}

def sex_to_int(x): return 1 if str(x).strip().lower()=="male" else 0
def area_to_int(x): return 1 if str(x).strip().lower()=="urban" else 0
def fault_to_int(x): return 1 if str(x).strip().lower()=="policy holder" else 0

DROP_COLS_ANYWAY = ["Combined Name","PolicyType","PolicyNumber","Unnamed: 33","Unnamed: 34","Statement"]

# Make encoder (loaded if available)
make_map: Dict[str, int] = {}

def _encode_make(col: pd.Series) -> pd.Series:
    # Use provided mapping; unknown -> -1
    return col.astype(str).apply(lambda v: make_map.get(v, -1)).astype(int)

def parse_range(val: str) -> float:
    if pd.isna(val):
        return 0.0
    val = str(val).strip().lower()
    if "less than" in val:
        nums = [int(s) for s in val.split() if s.isdigit()]
        return float(nums[0] - 2) if nums else 0.0
    if "more than" in val:
        nums = [int(s) for s in val.split() if s.isdigit()]
        return float(nums[0] + 2) if nums else 0.0
    if "to" in val:
        parts = [float(s) for s in val.replace("years","").replace("vehicle","").split() if s.replace('.','',1).isdigit()]
        if len(parts) == 2:
            return sum(parts)/2.0
    if val.replace('.','',1).isdigit():
        return float(val)
    if "vehicle" in val:
        nums = [int(s) for s in val.split() if s.isdigit()]
        return float(nums[0]) if nums else 1.0
    if "new" in val:
        return 0.5
    return 0.0

def engineer_policy_relation(row: pd.Series) -> int:
    policy_type = str(row.get("PolicyType", "")).strip()
    combined = f"{row.get('VehicleCategory','')} - {row.get('BasePolicy','')}"
    return 1 if policy_type == combined else 0

def preprocess_raw_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Derived policy relation if not present
    if "Policy realtion with Base" not in df.columns:
        df["Combined Name"] = df.get("VehicleCategory","").astype(str) + " - " + df.get("BasePolicy","").astype(str)
        df["Policy realtion with Base"] = df.apply(engineer_policy_relation, axis=1)

    # Binary / categorical to numeric
    if "Sex" in df.columns: df["Sex"] = df["Sex"].apply(sex_to_int).astype(int)
    if "AccidentArea" in df.columns: df["AccidentArea"] = df["AccidentArea"].apply(area_to_int).astype(int)
    if "Fault" in df.columns: df["Fault"] = df["Fault"].apply(fault_to_int).astype(int)
    if "PoliceReportFiled" in df.columns: df["PoliceReportFiled"] = df["PoliceReportFiled"].map(bool_map).astype(int)
    if "WitnessPresent" in df.columns: df["WitnessPresent"] = df["WitnessPresent"].map(bool_map).astype(int)
    if "AgentType" in df.columns: df["AgentType"] = df["AgentType"].map(agent_map).astype(int)
    if "VehicleCategory" in df.columns: df["VehicleCategory"] = df["VehicleCategory"].map(veh_map).astype(int)
    if "MaritalStatus" in df.columns: df["MaritalStatus"] = df["MaritalStatus"].map(marital_map).astype(int)
    if "NumberOfSuppliments" in df.columns: df["NumberOfSuppliments"] = df["NumberOfSuppliments"].map(sup_map).astype(float)
    if "BasePolicy" in df.columns: df["BasePolicy"] = df["BasePolicy"].map(base_policy_map).astype(int)

    for col in ["DayOfWeek","DayOfWeekClaimed"]:
        if col in df.columns:
            df[col] = df[col].replace(day_map).replace("0", 0).astype(int)
    for col in ["Month","MonthClaimed"]:
        if col in df.columns:
            df[col] = df[col].replace(month_map).astype(int)
    if "AddressChange_Claim" in df.columns:
        df["AddressChange_Claim"] = df["AddressChange_Claim"].replace(addr_change_map).astype(float)

    if "Make" in df.columns:
        df["Make"] = _encode_make(df["Make"])

    # Range-like text fields
    if "VehiclePrice" in df.columns: df["VehiclePrice"] = df["VehiclePrice"].apply(parse_range).astype(float)
    if "AgeOfVehicle" in df.columns: df["AgeOfVehicle"] = df["AgeOfVehicle"].apply(parse_range).astype(float)
    if "NumberOfCars" in df.columns: df["NumberOfCars"] = df["NumberOfCars"].apply(parse_range).astype(float)
    if "AgeOfPolicyHolder" in df.columns: df["AgeOfPolicyHolder"] = df["AgeOfPolicyHolder"].apply(parse_range).astype(float)

    # Drop known noise columns
    for c in DROP_COLS_ANYWAY:
        if c in df.columns: df = df.drop(columns=[c])
    for c in list(df.columns):
        if str(c).lower().startswith("unnamed:"):
            df = df.drop(columns=[c])

    # Any remaining object -> numeric best-effort
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(parse_range).astype(float)

    return df

# ---------- Model + features ----------
model = None
feature_cols: Optional[List[str]] = None

def _load_make_map_if_exists():
    global make_map
    if os.path.exists(MAKE_MAP_PATH):
        try:
            with open(MAKE_MAP_PATH, "r", encoding="utf-8") as f:
                make_map = json.load(f)
            if not isinstance(make_map, dict):
                make_map = {}
        except Exception:
            make_map = {}

def _load_feature_cols_from_file() -> Optional[List[str]]:
    if os.path.exists(FEATURE_COLUMNS_PATH):
        try:
            with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
                cols = json.load(f)
            if isinstance(cols, list) and all(isinstance(c, str) for c in cols):
                return cols
        except Exception:
            return None
    return None

@app.on_event("startup")
def _startup():
    global model, feature_cols

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}. Please include model.pkl in the image or set MODEL_PATH.")

    model = joblib.load(MODEL_PATH)
    if not hasattr(model, "predict"):
        raise RuntimeError("Loaded object does not look like a sklearn model with .predict().")

    # Preferred: models trained with DataFrame expose feature_names_in_
    feature_cols = list(getattr(model, "feature_names_in_", []))
    if not feature_cols:
        # Fallback to FEATURE_COLUMNS_PATH if provided
        loaded = _load_feature_cols_from_file()
        if loaded:
            feature_cols = loaded
        else:
            raise RuntimeError(
                "Unable to determine feature columns. "
                "Train your model with a pandas DataFrame so it has `feature_names_in_`, "
                "or provide FEATURE_COLUMNS_PATH (JSON array of column names)."
            )

    _load_make_map_if_exists()

# ---------- Request models ----------
class RawRow(RootModel[Dict[str, Any]]): ...
class NumericRow(RootModel[Dict[str, Any]]): ...

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

@app.post("/predict")
def predict_numeric(rows: List[NumericRow]):
    if not rows:
        raise HTTPException(status_code=400, detail="Request body must contain at least one row.")
    assert model is not None and feature_cols is not None

    df = pd.DataFrame([r.root for r in rows])

    # cleanup
    for c in list(df.columns):
        if str(c).lower().startswith("unnamed:"):
            df = df.drop(columns=[c])
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])

    # align to training features
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0
    df = df[feature_cols]

    preds_list = model.predict(df).tolist()
    probs_list = None
    try:
        probs_list = model.predict_proba(df)[:, 1].tolist()
    except Exception:
        pass

    labels = ["Fraud" if p == 1 else "Not Fraud" for p in preds_list]
    return {"predictions": labels, "probabilities_for_Fraud": probs_list}
