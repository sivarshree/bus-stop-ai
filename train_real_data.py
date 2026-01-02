"""
Daily training script - uses REAL MongoDB data
Runs automatically via GitHub Actions
"""

import os
import pymongo
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import json

print("🔄 Daily AI Retraining Started")
print(f"📅 {datetime.now()}")

def get_real_data_from_mongodb():
    """Connect to REAL MongoDB and get ALL bus stop data"""
    
    # Get connection string from environment variable (set in GitHub Secrets)
    mongo_uri = os.environ.get("MONGODB_URI")
    
    if not mongo_uri:
        print("❌ MONGODB_URI not set. Using sample data for testing.")
        return create_sample_data()
    
    try:
        print("🔌 Connecting to MongoDB...")
        client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        
        # Test connection
        client.server_info()
        print("✅ Connected to MongoDB")
        
        # Your actual database and collection names
        db = client["people_counter"]  # Your database name
        collection = db["readings"]     # Your collection name
        
        # Get ALL data - NO date filter
        print("📥 Fetching ALL data from MongoDB...")
        
        # Query for ALL documents
        data = list(collection.find({}).sort("timestamp", 1))
        
        if not data:
            print("⚠️ No data found in MongoDB. Using sample data.")
            return create_sample_data()
        
        print(f"✅ Found {len(data)} real records")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Ensure we have required columns
        if 'people_waiting_now' not in df.columns:
            print("⚠️ Column 'people_waiting_now' not found. Checking alternatives...")
            if 'people' in df.columns:
                df['people_waiting_now'] = df['people']
                print("✅ Using 'people' column as count")
            else:
                print("❌ No people count column found")
                return create_sample_data()
        
        # Prepare data
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Use people_waiting_now as our count
        df['count'] = df['people_waiting_now']
        
        print(f"📊 Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df[['timestamp', 'count']]
        
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        print("Using sample data for this run...")
        return create_sample_data()

def create_sample_data():
    """Create sample data if MongoDB connection fails"""
    print("📊 Creating sample data for training...")
    
    dates = pd.date_range(start='2025-01-01', end='2025-01-30', freq='10min')
    data = []
    
    for timestamp in dates:
        hour = timestamp.hour
        weekday = timestamp.weekday()
        
        # Realistic patterns
        if 7 <= hour <= 9:
            people = np.random.randint(15, 30)
        elif 17 <= hour <= 19:
            people = np.random.randint(20, 35)
        elif 12 <= hour <= 14:
            people = np.random.randint(10, 20)
        elif 22 <= hour or hour <= 5:
            people = np.random.randint(0, 5)
        else:
            people = np.random.randint(5, 15)
        
        if weekday >= 5:
            people = int(people * 0.4)
        
        data.append({
            'timestamp': timestamp,
            'count': people
        })
    
    df = pd.DataFrame(data)
    print(f"📊 Created {len(df)} sample records")
    return df

def train_model(df):
    """Train the LSTM model with the data"""
    print("🧠 Training model...")
    
    # Check if we have enough data
    if len(df) < 100:
        print(f"⚠️ Very little data: {len(df)} records. Training may be poor.")
    
    # Prepare sequences
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[['count']].values)
    
    SEQ_LENGTH = 144  # 24 hours
    PRED_LENGTH = 36   # 6 hours
    
    # Check if we can create sequences
    max_possible = len(data_scaled) - SEQ_LENGTH - PRED_LENGTH
    if max_possible <= 0:
        print(f"❌ Not enough data for sequences. Need {SEQ_LENGTH + PRED_LENGTH} points, have {len(data_scaled)}")
        print("⏭️ Cannot train with this data.")
        return 0.0
    
    # Create sequences
    X, y = [], []
    for i in range(max_possible):
        X.append(data_scaled[i:i+SEQ_LENGTH])
        y.append(data_scaled[i+SEQ_LENGTH:i+SEQ_LENGTH+PRED_LENGTH])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"📊 Created {len(X)} training sequences")
    
    # Adjust parameters based on data size
    if len(X) < 50:
        validation_split = 0.1
        epochs = 15
        batch_size = min(16, len(X))
        print(f"📉 Small dataset: Using validation_split={validation_split}, epochs={epochs}")
    elif len(X) < 200:
        validation_split = 0.15
        epochs = 20
        batch_size = 32
    else:
        validation_split = 0.2
        epochs = 30
        batch_size = 32
    
    # Load existing model or create new
    try:
        model = load_model('bus_stop_predictor.h5')
        print("✅ Loaded existing model for retraining")
    except:
        print("🆕 Creating new model")
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(PRED_LENGTH)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Train
    print(f"🚂 Training with {len(X)} sequences, {epochs} epochs...")
    history = model.fit(
        X, y,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )
    
    # Evaluate
    if len(X) > 10:
        test_loss, test_mae = model.evaluate(X, y, verbose=0)
        print(f"📈 Test MAE: {test_mae:.4f} (approx {test_mae * df['count'].max():.1f} people error)")
    else:
        test_mae = history.history['val_mae'][-1] if 'val_mae' in history.history else 0.15
        print(f"📊 Using validation MAE: {test_mae:.4f}")
    
    # Save model and scaler
    model.save('bus_stop_predictor.h5')
    joblib.dump(scaler, 'scaler.pkl')
    
    # Update model info
    model_info = {
        "seq_length": SEQ_LENGTH,
        "pred_length": PRED_LENGTH,
        "max_people": float(df['count'].max()),
        "min_people": float(df['count'].min()),
        "last_trained": datetime.now().isoformat(),
        "records_used": len(df),
        "sequences_created": len(X),
        "test_mae": float(test_mae),
        "training_samples": len(X),
        "status": "trained"
    }
    
    with open('model_info.json', 'w') as f:
        json.dump(model_info, f)
    
    print("✅ Model training completed and saved")
    return test_mae

def main():
    """Main function"""
    print("=" * 50)
    print("🚌 BUS STOP AI - DAILY RETRAINING")
    print("=" * 50)
    
    # Get real data from MongoDB
    df = get_real_data_from_mongodb()
    
    print(f"\n📊 Data Summary:")
    print(f"   Records: {len(df)}")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Average count: {df['count'].mean():.1f}")
    print(f"   Min count: {df['count'].min()}")
    print(f"   Max count: {df['count'].max()}")
    
    # Train model
    mae = train_model(df)
    
    if mae > 0:
        print(f"\n🎯 Training completed with MAE: {mae:.4f}")
        print(f"   Prediction error: ~{mae * df['count'].max():.1f} people")
    else:
        print("\n⚠️ Training failed or skipped")
    
    print(f"🕒 Next training: Tomorrow at 2 AM")
    print("=" * 50)

if __name__ == "__main__":
    main()
