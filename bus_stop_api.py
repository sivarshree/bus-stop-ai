# bus_stop_api.py
# FastAPI server for bus stop predictions

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import json
from datetime import datetime, timedelta
from typing import List
import os
from pymongo import MongoClient

# Initialize FastAPI
app = FastAPI(title="Bus Stop Prediction API")

# Load model and components
try:
    model = load_model("bus_stop_predictor.h5")
    scaler = joblib.load("scaler.pkl")
    
    with open("model_info.json", "r") as f:
        model_info = json.load(f)
    
    SEQ_LENGTH = model_info["seq_length"]
    PRED_LENGTH = model_info["pred_length"]
    
    print(f"✅ Model loaded. Sequence: {SEQ_LENGTH}, Predict: {PRED_LENGTH}")
    
    # MongoDB connection
    print("🔌 Attempting MongoDB connection...")
    MONGODB_URI = os.environ.get("MONGODB_URI")
    
    if MONGODB_URI:
        try:
            # Hide password in logs
            hidden_uri = MONGODB_URI.split('@')[0].split(':')[0] + ':***@' + MONGODB_URI.split('@')[1]
            print(f"✅ MongoDB URI found: {hidden_uri}")
            
            mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
            mongo_client.server_info()  # Test connection
            mongo_db = mongo_client["people_counter"]
            mongo_collection = mongo_db["readings"]
            print("✅ Connected to MongoDB successfully!")
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            mongo_collection = None
    else:
        print("⚠️ MONGODB_URI not found in environment variables")
        mongo_collection = None
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

# Request/Response models
class PredictionRequest(BaseModel):
    historical_data: List[float]  # Past 24 hours of people counts (144 values)
    location: str = "bus_stop_1"  # Optional: bus stop ID

class PredictionResponse(BaseModel):
    predictions: List[float]      # Next 6 hours of predictions (36 values)
    timestamps: List[str]         # Corresponding timestamps
    generated_at: str
    location: str

@app.get("/")
def root():
    return {
        "message": "Bus Stop Prediction API",
        "endpoints": {
            "/predict": "POST - Get predictions for next 6 hours",
            "/health": "GET - Check API health"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Predict next 6 hours of bus stop occupancy."""
    
    # Validate input length
    if len(request.historical_data) < SEQ_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {SEQ_LENGTH} historical values, got {len(request.historical_data)}"
        )
    
    try:
        # Prepare data
        hist_array = np.array(request.historical_data[-SEQ_LENGTH:]).reshape(-1, 1)
        hist_scaled = scaler.transform(hist_array)
        hist_scaled = hist_scaled.reshape(1, SEQ_LENGTH, 1)
        
        # Make prediction
        pred_scaled = model.predict(hist_scaled, verbose=0)
        pred_actual = scaler.inverse_transform(pred_scaled.reshape(-1, 1))
        predictions = pred_actual.flatten().tolist()
        
        # Generate timestamps (every 10 minutes for next 6 hours)
        now = datetime.now()
        timestamps = [
            (now + timedelta(minutes=(i+1)*10)).strftime("%H:%M")
            for i in range(PRED_LENGTH)
        ]
        
        return PredictionResponse(
            predictions=predictions,
            timestamps=timestamps,
            generated_at=now.isoformat(),
            location=request.location
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


