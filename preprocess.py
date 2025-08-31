# preprocess.py — shared encoders + preprocess_raw_df
from typing import Dict, Any, Optional, List
import pandas as pd

# ---- Encoders / maps ----
day_map = {"Sunday":0,"Monday":1,"Tuesday":2,"Wednesday":3,"Thursday":4,"Friday":5,"Saturday":6}
month_map = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
bool_map = {"No":0, "Yes":1}
agent_map = {"External":0, "Internal":1}
veh_map = {"Sport":0, "Sedan":1, "Utility":2}
marital_map = {"Widow":0, "Single":1, "Married":2, "Divorced":3}
sup_map = {"none":0, "1 to 2":1.5, "3 to 5":4, "more than 5":6}
base_policy_map = {"Liability":0, "Collision":1, "All Perils":2}
addr_change_map = {"1 year":1, "no change":0, "4 to 8 years":6, "2 to 3 years":2.5, "under 6 months":0.3}

DROP_COLS_ANYWAY = ["Combined Name","PolicyType","PolicyNumber","Unnamed: 33","Unnamed: 34","Statement"]

# Global mapping for Make (built from training data)
make_map: Dict[str, int] = {}

def _build_make_map(series: pd.Series) -> Dict[str, int]:
    labels = sorted(set(str(v) for v in series.dropna().tolist()))
    return {lab: i for i, lab in enumerate(labels)}

def _encode_make(col: pd.Series) -> pd.Series:
    return col.apply(lambda v: make_map.get(str(v), -1)).astype(int)

def sex_to_int(x): return 1 if str(x).strip().lower()=="male" else 0
def area_to_int(x): return 1 if str(x).strip().lower()=="urban" else 0
def fault_to_int(x): return 1 if str(x).strip().lower()=="policy holder" else 0

def parse_range(val: str) -> float:
    import pandas as pd
    if pd.isna(val):
        return 0.0
    val = str(val).strip().lower()

    if "less than" in val:
        nums = [int(s) for s in val.split() if s.isdigit()]
        return nums[0] - 2 if nums else 0.0
    if "more than" in val:
        nums = [int(s) for s in val.split() if s.isdigit()]
        return nums[0] + 2 if nums else 0.0
    if "to" in val:
        parts = [float(s) for s in val.replace("years","").replace("vehicle","").split() if s.replace('.','',1).isdigit()]
        if len(parts) == 2:
            return sum(parts)/2
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

    # Policy relation engineer (kept for compatibility; target is not used at inference)
    if "Policy realtion with Base" not in df.columns:
        df["Combined Name"] = df.get("VehicleCategory","").astype(str) + " - " + df.get("BasePolicy","").astype(str)
        df["Policy realtion with Base"] = df.apply(engineer_policy_relation, axis=1)

    # Encode categoricals
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].apply(sex_to_int).astype(int)
    if "AccidentArea" in df.columns:
        df["AccidentArea"] = df["AccidentArea"].apply(area_to_int).astype(int)
    if "Fault" in df.columns:
        df["Fault"] = df["Fault"].apply(fault_to_int).astype(int)
    if "PoliceReportFiled" in df.columns:
        df["PoliceReportFiled"] = df["PoliceReportFiled"].map(bool_map).astype(int)
    if "WitnessPresent" in df.columns:
        df["WitnessPresent"] = df["WitnessPresent"].map(bool_map).astype(int)
    if "AgentType" in df.columns:
        df["AgentType"] = df["AgentType"].map(agent_map).astype(int)
    if "VehicleCategory" in df.columns:
        df["VehicleCategory"] = df["VehicleCategory"].map(veh_map).astype(int)
    if "MaritalStatus" in df.columns:
        df["MaritalStatus"] = df["MaritalStatus"].map(marital_map).astype(int)
    if "NumberOfSuppliments" in df.columns:
        df["NumberOfSuppliments"] = df["NumberOfSuppliments"].map(sup_map).astype(float)
    if "BasePolicy" in df.columns:
        df["BasePolicy"] = df["BasePolicy"].map(base_policy_map).astype(int)

    for col in ["DayOfWeek","DayOfWeekClaimed"]:
        if col in df.columns:
            df[col] = df[col].replace(day_map).replace("0", 0).astype(int)
    for col in ["Month","MonthClaimed"]:
        if col in df.columns:
            df[col] = df[col].replace(month_map).astype(int)
    if "AddressChange_Claim" in df.columns:
        df["AddressChange_Claim"] = df["AddressChange_Claim"].replace(addr_change_map).astype(float)

    if "Make" in df.columns:
        df["Make"] = _encode_make(df["Make"].astype(str))

    # String ranges → numeric
    if "VehiclePrice" in df.columns:
        df["VehiclePrice"] = df["VehiclePrice"].apply(parse_range).astype(float)
    if "AgeOfVehicle" in df.columns:
        df["AgeOfVehicle"] = df["AgeOfVehicle"].apply(parse_range).astype(float)
    if "NumberOfCars" in df.columns:
        df["NumberOfCars"] = df["NumberOfCars"].apply(parse_range).astype(float)
    if "AgeOfPolicyHolder" in df.columns:
        df["AgeOfPolicyHolder"] = df["AgeOfPolicyHolder"].apply(parse_range).astype(float)

    # Drop junk
    for c in DROP_COLS_ANYWAY:
        if c in df.columns:
            df = df.drop(columns=[c])
    for c in list(df.columns):
        if str(c).lower().startswith("unnamed:"):
            df = df.drop(columns=[c])

    # Final safeguard: any leftover object → parse_range
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(parse_range).astype(float)

    return df
