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
    """Connect to REAL MongoDB and get bus stop data"""
    
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
        
        # Change these to your actual database/collection names
        db = client["people_counter"]  # Your database name
        collection = db["readings"]  # Your collection name
        
        # Get last 60 days of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        
        print(f"📥 Fetching data from {start_date.date()} to {end_date.date()}")
        
        # Query MongoDB for real data
        query = {
            "timestamp": {"$gte": start_date, "$lte": end_date}
        }
        
        data = list(collection.find(query).sort("timestamp", 1))
        
        if not data:
            print("⚠️ No data found in MongoDB. Using sample data.")
            return create_sample_data()
        
        print(f"✅ Found {len(data)} real records")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Ensure we have required columns
        if 'people_waiting_now' not in df.columns:
            print("⚠️ Column 'people_waiting_now' not found. Using 'people' or sample data.")
            if 'people' in df.columns:
                df['people_waiting_now'] = df['people']
            else:
                return create_sample_data()
        
        # Prepare data
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Use people_waiting_now as our count
        df['count'] = df['people_waiting_now']
        
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
    
    # Prepare sequences (same as before)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[['count']].values)
    
    SEQ_LENGTH = 144  # 24 hours
    PRED_LENGTH = 36   # 6 hours
    
    # Create sequences
    X, y = [], []
    for i in range(len(data_scaled) - SEQ_LENGTH - PRED_LENGTH):
        X.append(data_scaled[i:i+SEQ_LENGTH])
        y.append(data_scaled[i+SEQ_LENGTH:i+SEQ_LENGTH+PRED_LENGTH])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"📊 Training on {len(X_train)} sequences")
    
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
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=32,
        verbose=0
    )
    
    # Evaluate
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"📈 Test MAE: {test_mae:.4f}")
    
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
        "test_mae": float(test_mae)
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
    
    # Train model
    mae = train_model(df)
    
    print(f"\n🎯 Training completed with MAE: {mae:.4f}")
    print(f"🕒 Next training: Tomorrow at 2 AM")
    print("=" * 50)

if __name__ == "__main__":
    main()
