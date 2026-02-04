import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = "models/model.joblib"

app = FastAPI(title="Trip Duration Prediction API")

model = joblib.load(MODEL_PATH)

class TripInput(BaseModel):
    trip_distance: float
    passenger_count: int
    pickup_hour: int
    pickup_dayofweek: int

@app.post("/predict")
def predict(input: TripInput):
    data = pd.DataFrame([input.dict()])
    prediction = model.predict(data)[0]
    return {"predicted_trip_duration_minutes": round(float(prediction), 2)}
