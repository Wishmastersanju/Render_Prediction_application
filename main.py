from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pickle

# Load the XGBoost model
try:
    with open("xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Initialize FastAPI
app = FastAPI()

# Allow CORS (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the expected input format
class ModelInput(BaseModel):
    age: float
    weight: float

# Health check route
@app.get("/")
def read_root():
    return {"message": "Hydration predictor API is running"}

# Prediction endpoint
@app.post("/predict")
def predict(input: ModelInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Prepare features
        features = np.array([[input.age, input.weight]])
        prediction = model.predict(features)
        return {"prediction": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
