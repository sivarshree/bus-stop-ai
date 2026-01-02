# Add these imports at the top
from pymongo import MongoClient
import os
from datetime import datetime, timedelta

# Add MongoDB connection (after model loading)
try:
    MONGODB_URI = os.environ.get("MONGODB_URI", "your_mongodb_uri_here")
    mongo_client = MongoClient(MONGODB_URI)
    mongo_db = mongo_client["people_counter"]
    mongo_collection = mongo_db["readings"]
    print("✅ Connected to MongoDB")
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    mongo_collection = None

# Add new endpoint to get current occupancy
@app.get("/current/{bus_stop_id}")
def get_current_occupancy(bus_stop_id: str):
    """Get latest people count for a bus stop"""
    if not mongo_collection:
        raise HTTPException(status_code=500, detail="MongoDB not connected")
    
    try:
        # Get the latest reading for this bus stop
        latest_data = mongo_collection.find_one(
            {"location": bus_stop_id},
            sort=[("timestamp", -1)]  # Get most recent
        )
        
        if not latest_data:
            return {
                "bus_stop_id": bus_stop_id,
                "people_waiting_now": 0,
                "timestamp": datetime.now().isoformat(),
                "has_data": False,
                "message": "No data available"
            }
        
        return {
            "bus_stop_id": bus_stop_id,
            "people_waiting_now": latest_data.get("people_waiting_now", 0),
            "interval_total": latest_data.get("interval_total", 0),
            "timestamp": latest_data.get("timestamp").isoformat() if latest_data.get("timestamp") else datetime.now().isoformat(),
            "source": latest_data.get("source", "unknown"),
            "has_data": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MongoDB error: {str(e)}")

# Add endpoint to get historical data for AI predictions
@app.get("/historical/{bus_stop_id}")
def get_historical_data(bus_stop_id: str, hours: int = 24):
    """Get historical people counts for AI prediction input"""
    if not mongo_collection:
        raise HTTPException(status_code=500, detail="MongoDB not connected")
    
    try:
        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        # Get data for the last X hours
        historical_data = list(mongo_collection.find({
            "location": bus_stop_id,
            "timestamp": {"$gte": start_time, "$lte": end_time}
        }).sort("timestamp", 1))  # Oldest first
        
        if not historical_data:
            return {
                "bus_stop_id": bus_stop_id,
                "data": [],
                "count": 0,
                "message": "No historical data available"
            }
        
        # Format data for AI input (every 10 minutes)
        formatted_data = []
        for record in historical_data:
            formatted_data.append({
                "timestamp": record.get("timestamp").isoformat() if record.get("timestamp") else None,
                "people": record.get("people_waiting_now", 0),
                "interval_total": record.get("interval_total", 0)
            })
        
        # Extract just the people counts for AI prediction
        people_counts = [record.get("people_waiting_now", 0) for record in historical_data]
        
        return {
            "bus_stop_id": bus_stop_id,
            "data": formatted_data,
            "people_counts": people_counts,
            "count": len(historical_data),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "has_data": len(historical_data) > 0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MongoDB error: {str(e)}")
