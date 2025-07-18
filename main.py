# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pickle

# Load the XGBoost model
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, use your app’s domain instead of "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input data format
class ModelInput(BaseModel):
    feature1: float
    feature2: float

@app.get("/")
def read_root():
    return {"message": "ML model API is running"}

@app.post("/predict")
def predict(input: ModelInput):
    # Convert input to NumPy array
    features = np.array([[input.feature1, input.feature2]])
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}
