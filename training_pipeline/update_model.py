import sqlite3
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "..", "models", "ecoForecast.db")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "weather_lstm_model.keras")

FEATURES = ['temperature', 'humidity', 'pressure', 'wind_speed', 'aqi', 'weather_code']

def map_wmo_to_class(code):
    if pd.isna(code): return 1
    code = int(code)
    if code == 0: return 0
    if code in [1, 2, 3, 45, 48]: return 1
    if 51 <= code <= 82: return 2
    if code >= 95: return 3
    return 1

def fine_tune_recent_data():
    print("🔄 Starting Incremental Fine-Tuning for Dual Model...")

    conn = sqlite3.connect(DB_PATH)
    query = f"SELECT {', '.join(FEATURES)} FROM weather_data ORDER BY timestamp DESC LIMIT 1000"
    df = pd.read_sql(query, conn)
    conn.close()

    if len(df) < 50:
        print("Not enough data to fine-tune yet.")
        return

    df = df.iloc[::-1]
    df.fillna(0, inplace=True)
    df['weather_class'] = df['weather_code'].apply(map_wmo_to_class)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[FEATURES].values)
    weather_classes = df['weather_class'].values

    look_back = 24
    X, y_temp, y_weather = [], [], []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i: i + look_back])
        y_temp.append(scaled_data[i + look_back, 0])
        y_weather.append(weather_classes[i + look_back])

    X = np.array(X)
    y_temp = np.array(y_temp)
    y_weather = np.array(y_weather)

    model = load_model(MODEL_PATH)

    # Recompile with our two different loss functions
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss={'temp_out': 'mse', 'weather_out': 'sparse_categorical_crossentropy'}
    )

    print("🏃 Learning from recent weather patterns...")
    model.fit(
        X,
        {'temp_out': y_temp, 'weather_out': y_weather},
        epochs=2, batch_size=32, verbose=1
    )

    model.save(MODEL_PATH)
    print(f"✅ Model updated and saved to {MODEL_PATH}")

if __name__ == "__main__":
    fine_tune_recent_data()