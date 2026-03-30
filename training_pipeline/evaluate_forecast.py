import sqlite3
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# --- Path Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "..", "models", "ecoForecast.db")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "weather_lstm_model.keras")

# Our new exactly 6 features
FEATURES = ['temperature', 'humidity', 'pressure', 'wind_speed', 'aqi', 'weather_code']


def evaluate_all_cities():
    conn = sqlite3.connect(DB_PATH)
    cities_df = pd.read_sql("SELECT name FROM locations", conn)
    model = load_model(MODEL_PATH)

    results_summary = []

    print(f"\n🚀 Starting Dual-Model Evaluation for {len(cities_df)} Malaysian Cities...")
    print("-" * 60)

    for city_name in cities_df['name']:
        # Fetch the exact 6 features the AI was trained on
        query = f"""
        SELECT w.timestamp, w.temperature, w.humidity, w.pressure, w.wind_speed, w.aqi, w.weather_code 
        FROM weather_data w
        JOIN locations l ON w.location_id = l.id
        WHERE l.name = '{city_name}'
        ORDER BY w.timestamp ASC LIMIT 300
        """
        df = pd.read_sql(query, conn)

        if df.empty or len(df) < 50:
            continue

        # Clean missing values
        df.fillna(0, inplace=True)

        # Drop timestamp for scaling
        df_features = df[FEATURES]

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df_features)

        look_back = 24
        X, actuals_scaled = [], []
        for i in range(len(scaled_data) - look_back):
            X.append(scaled_data[i: i + look_back])
            # Index 0 is Temperature
            actuals_scaled.append(scaled_data[i + look_back, 0])

        # THE NEW WAY: Predict gives us TWO outputs now!
        predictions = model.predict(np.array(X), verbose=0)

        # Branch A is Temperature (Index 0), Branch B is Weather Class (Index 1)
        predictions_temp_scaled = predictions[0]

        # Denormalize back to Celsius
        dummy = np.zeros((len(predictions_temp_scaled), len(FEATURES)))

        # Denormalize predicted temp
        dummy[:, 0] = predictions_temp_scaled.flatten()
        pred_c = scaler.inverse_transform(dummy)[:, 0]

        # Denormalize actual temp
        dummy[:, 0] = np.array(actuals_scaled).flatten()
        act_c = scaler.inverse_transform(dummy)[:, 0]

        # Calculate Average Error for this city
        mae = np.mean(np.abs(act_c - pred_c))
        results_summary.append({'City': city_name, 'Avg Error': mae})
        print(f"✅ {city_name:<15} | Avg Temp Error: {mae:.2f}°C")

    conn.close()

    summary_df = pd.DataFrame(results_summary).sort_values(by='Avg Error')
    print("\n🏆 --- CITY ACCURACY LEADERBOARD ---")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    evaluate_all_cities()