from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import List

# Load the model
model = joblib.load("model.pkl")

# Define input schema
class InsuranceFeatures(BaseModel):
    Month: int
    WeekOfMonth: int
    DayOfWeek: int
    Make: int
    AccidentArea: int
    DayOfWeekClaimed: int
    MonthClaimed: int
    WeekOfMonthClaimed: int
    Sex: int
    MaritalStatus: int
    Fault: int
    VehicleCategory: int
    VehiclePrice: int
    PolicyType: int
    Policy: int
    DriverRating: int
    Days_Policy_Accident: int
    Days_Policy_Claim: int
    PastNumberOfClaims: int
    AgeOfVehicle: int
    AgeOfPolicyHolder: float
    PoliceReportFiled: int
    WitnessPresent: int
    AgentType: int
    NumberOfSuppliments: float
    AddressChange_Claim: float
    NumberOfCars: float
    Year: int
    BasePolicy: int
    Policy_relation_with_Base: int

app = FastAPI()

@app.post("/predict")
def predict(features: List[InsuranceFeatures]):
    # Convert input into dataframe
    df = pd.DataFrame([f.dict() for f in features])
    preds = model.predict(df)
    return {"predictions": preds.tolist()}
