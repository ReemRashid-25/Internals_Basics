from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("models/tuned_model.pkl")

class InputData(BaseModel):
    trip_km: float = Field(..., ge=1, le=100)
    passenger_count: int = Field(..., ge=1, le=6)
    is_surge: int = Field(..., ge=0, le=1)
    traffic_index: int = Field(..., ge=1, le=5)

@app.get("/heartbeat")
def heartbeat():
    return {"alive": True, "service": "UrbanRide fare_amount API"}

@app.post("/infer")
def infer(data: InputData):
    try:
        features = np.array([[ 
            data.trip_km,
            data.passenger_count,
            data.is_surge,
            data.traffic_index
        ]])
        pred = model.predict(features)[0]
        return {"prediction": float(pred)}
    except:
        raise HTTPException(status_code=422, detail="Invalid input")
