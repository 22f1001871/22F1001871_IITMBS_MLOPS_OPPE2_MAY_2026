from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime, timezone

app = FastAPI()

# Load trained model
model = joblib.load("model/heart_disease_model.joblib")

# Exact feature order used by the model
FEATURES = [
    "sno",
    "age",
    "gender",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]

# Define request schema
class PatientData(BaseModel):
    sno: int
    age: int
    gender: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: PatientData):
    # Convert to DataFrame with correct feature order
    input_data = pd.DataFrame([[getattr(data, f) for f in FEATURES]], columns=FEATURES)

    # Make prediction
    prediction = model.predict(input_data)[0]

    probability = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        classes = model.classes_
        predicted_index = list(classes).index(prediction)
        probability = float(probabilities[predicted_index])

    timestamp = datetime.now(timezone.utc).isoformat()

    # Log to stdout
    print({
        "timestamp": timestamp,
        "input": data.dict(),
        "prediction": str(prediction),
        "probability": probability
    }, flush=True)

    return {
        "prediction": str(prediction),
        "probability": probability,
        "timestamp": timestamp
    }
