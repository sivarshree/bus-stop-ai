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

# Global MongoDB collection
mongo_collection = None

# --- TIMEZONE HELPER FUNCTION ---
def to_malaysia_time(dt_obj=None):
    """
    Converts a UTC datetime object (or current UTC time) to Malaysia Time (UTC+8).
    If no time is provided, returns current Malaysia time.
    """
    if dt_obj is None:
        dt_obj = datetime.utcnow()
    
    # Add 8 hours to the UTC time
    myt_time = dt_obj + timedelta(hours=8)
    return myt_time

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
            "/health": "GET - Check API health",
            "/current/{bus_stop_id}": "GET - Get current occupancy",
            "/historical/{bus_stop_id}": "GET - Get historical data"
        }
    }

@app.get("/health")
def health_check():
    mongodb_status = "connected" if mongo_collection else "disconnected"
    return {
        "status": "healthy",
        "model_loaded": True,
        "mongodb": mongodb_status,
        "timestamp_myt": to_malaysia_time().isoformat() # Updated to MYT
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
        # UPDATED: Use Malaysia Time as base
        now_myt = to_malaysia_time()
        
        timestamps = [
            (now_myt + timedelta(minutes=(i+1)*10)).strftime("%H:%M")
            for i in range(PRED_LENGTH)
        ]
        
        return PredictionResponse(
            predictions=predictions,
            timestamps=timestamps,
            generated_at=now_myt.isoformat(),
            location=request.location
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ===== MongoDB ENDPOINTS (UPDATED) =====

@app.get("/current/{bus_stop_id}")
async def get_current_occupancy(bus_stop_id: str, source: str = "raspberry_pi_01"):
    """
    Get latest people count for a bus stop from MongoDB.
    Defaults to source='raspberry_pi_01' to ignore synthetic data.
    """
    try:
        if mongo_collection is None:
            return {
                "bus_stop_id": bus_stop_id,
                "people_waiting_now": 0,
                "timestamp": to_malaysia_time().isoformat(),
                "has_data": False,
                "message": "MongoDB not connected"
            }
        
        # Get latest reading specifically from the requested source (defaults to Pi)
        latest = mongo_collection.find_one(
            {
                "location": bus_stop_id,
                "source": source
            },
            sort=[("timestamp", -1)]
        )
        
        if not latest:
            return {
                "bus_stop_id": bus_stop_id,
                "people_waiting_now": 0,
                "timestamp": to_malaysia_time().isoformat(),
                "has_data": False,
                "message": f"No data found for source: {source}"
            }
        
        # Convert timestamp to Malaysia Time
        timestamp = latest.get("timestamp")
        if isinstance(timestamp, datetime):
            # Convert the stored UTC time to MYT
            timestamp = to_malaysia_time(timestamp).isoformat()
        else:
            timestamp = to_malaysia_time().isoformat()
        
        return {
            "bus_stop_id": bus_stop_id,
            "people_waiting_now": latest.get("people_waiting_now", 0),
            "interval_total": latest.get("interval_total", 0),
            "timestamp": timestamp, # This is now MYT
            "source": latest.get("source", "unknown"),
            "has_data": True
        }
        
    except Exception as e:
        return {
            "bus_stop_id": bus_stop_id,
            "people_waiting_now": 0,
            "timestamp": to_malaysia_time().isoformat(),
            "has_data": False,
            "message": f"Error: {str(e)}"
        }

@app.get("/historical/{bus_stop_id}")
async def get_historical_data(bus_stop_id: str, hours: int = 24, source: str = "raspberry_pi_01"):
    """
    Get historical data for AI predictions.
    Defaults to source='raspberry_pi_01' so graphs don't mix real/synthetic data.
    """
    try:
        if mongo_collection is None:
            return {
                "bus_stop_id": bus_stop_id,
                "data": [],
                "people_counts": [],
                "count": 0,
                "has_data": False,
                "message": "MongoDB not connected"
            }
        
        # Get data for last X hours (Query still uses UTC for Mongo)
        end_time_utc = datetime.utcnow()
        start_time_utc = end_time_utc - timedelta(hours=hours)
        
        cursor = mongo_collection.find({
            "location": bus_stop_id,
            "source": source,
            "timestamp": {"$gte": start_time_utc, "$lte": end_time_utc}
        }).sort("timestamp", 1)
        
        data = list(cursor)
        
        if not data:
            return {
                "bus_stop_id": bus_stop_id,
                "data": [],
                "people_counts": [],
                "count": 0,
                "has_data": False,
                "message": f"No historical data for source: {source}"
            }
        
        # Extract just the counts for AI
        people_counts = [record.get("people_waiting_now", 0) for record in data]
        
        # Format data for response with Malaysia Time
        formatted_data = []
        for record in data:
            timestamp = record.get("timestamp")
            if isinstance(timestamp, datetime):
                # Convert to MYT for display
                timestamp = to_malaysia_time(timestamp).isoformat()
            
            formatted_data.append({
                "timestamp": timestamp, # MYT Time
                "people_waiting_now": record.get("people_waiting_now", 0),
                "interval_total": record.get("interval_total", 0),
                "source": record.get("source", "unknown")
            })
        
        return {
            "bus_stop_id": bus_stop_id,
            "data": formatted_data,
            "people_counts": people_counts,
            "count": len(data),
            "has_data": True,
            "start_time": to_malaysia_time(start_time_utc).isoformat(),
            "end_time": to_malaysia_time(end_time_utc).isoformat()
        }
        
    except Exception as e:
        return {
            "bus_stop_id": bus_stop_id,
            "data": [],
            "people_counts": [],
            "count": 0,
            "has_data": False,
            "message": f"Error: {str(e)}"
        }
