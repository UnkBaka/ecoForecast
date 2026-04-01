import numpy as np
import tensorflow as tf
import pandas as pd
import requests
import time
import sqlite3
import os
import sys
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

from generate_map import create_dashboard_map

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "models", "ecoForecast.db")
MODEL_PATH = os.path.join(BASE_DIR, "models", "weather_lstm_model.keras")

model = None
FEATURES = ['temperature', 'humidity', 'pressure', 'wind_speed', 'aqi', 'weather_code']


def load_ai_model():
    global model
    if os.path.exists(MODEL_PATH):
        print(f"Loading Multi-Output AI Model...")
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print("✅ Dual-Brain Model loaded successfully.")
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
    else:
        print(f"❌ Model file not found at {MODEL_PATH}")


def get_connection():
    return sqlite3.connect(DB_PATH)


def get_forecast_by_name(city_name, target_time=None):
    global model
    if model is None:
        return None, None, 0.0

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM locations WHERE name LIKE ?", (f"%{city_name}%",))
    row = cur.fetchone()

    if not row:
        conn.close()
        return None, None, 0.0

    loc_id = row[0]
    target_str = (target_time or datetime.now()).strftime('%Y-%m-%d %H:%M:%S')

    query = f"SELECT {', '.join(FEATURES)} FROM weather_data WHERE location_id = ? AND timestamp <= ? ORDER BY timestamp DESC LIMIT 24"
    df = pd.read_sql(query, conn, params=(loc_id, target_str))
    conn.close()

    if len(df) < 24:
        return None, None, 0.0

    df_sorted = df.iloc[::-1].reset_index(drop=True)

    # --- FIXED SCALING LOGIC ---
    # Instead of fit_transforming only 24 hours, we use a fixed range
    # to represent the typical Malaysian climate variance.
    data_values = df_sorted[FEATURES].values

    # Manual Scaling to [0, 1] based on expected Min/Max
    # Temp: 20-40, Hum: 30-100, Press: 990-1020, Wind: 0-30, AQI: 0-200, Code: 0-100
    mins = np.array([20.0, 30.0, 990.0, 0.0, 0.0, 0.0])
    maxs = np.array([40.0, 100.0, 1020.0, 30.0, 200.0, 100.0])

    scaled_data = (data_values - mins) / (maxs - mins)
    input_data = scaled_data.reshape(1, 24, len(FEATURES))

    predictions = model.predict(input_data, verbose=0)

    # Inverse scaling for the Temperature output
    # predictions[0][0,0] is the 0-1 value for temp
    final_temp = round(float(predictions[0][0, 0] * (maxs[0] - mins[0]) + mins[0]), 2)

    # Weather Class
    probs = predictions[1][0]
    rain_chance = round(float(max(probs[2], probs[3]) * 100), 1)
    ai_class = np.argmax(probs)
    wmo_translation = {0: 0, 1: 2, 2: 61, 3: 95}

    return final_temp, wmo_translation.get(ai_class, 2), rain_chance


def heal_missing_data(conn):
    cur = conn.cursor()
    cur.execute("SELECT id, name, lat, lon FROM locations")
    locations = cur.fetchall()

    print(f"\n[*] 🛠️ Healing data gaps and generating future forecast...")

    for loc_id, name, lat, lon in locations:
        # --- PART 1: HEAL HISTORY (API) ---
        w_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,surface_pressure,windspeed_10m,weathercode&timezone=auto&past_days=8"
        try:
            res = requests.get(w_url).json().get('hourly', {})
            if res:
                print(f"  [>] Syncing history for {name}...")
                for i, api_time in enumerate(res.get('time', [])):
                    db_time = api_time.replace('T', ' ') + ":00"

                    # Fill Weather Table
                    cur.execute("SELECT id FROM weather_data WHERE location_id = ? AND timestamp = ?",
                                (loc_id, db_time))
                    if not cur.fetchone():
                        cur.execute("""
                            INSERT INTO weather_data (location_id, timestamp, temperature, humidity, pressure, wind_speed, weather_code, aqi)
                            VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                        """, (loc_id, db_time, res['temperature_2m'][i], res['relative_humidity_2m'][i],
                              res['surface_pressure'][i], res['windspeed_10m'][i], res['weathercode'][i]))
                conn.commit()
        except Exception as e:
            print(f"  [-] Error syncing history for {name}: {e}")

        # --- PART 2: GENERATE FUTURE 24H (LSTM AI) ---
        # Move this OUTSIDE the try/except block so it always runs
        print(f"  [>] Generating future 24h forecast for {name}...")
        now = datetime.now().replace(minute=0, second=0, microsecond=0)

        for h in range(1, 25):
            # 2. Add 'h' hours to NOW (1h, 2h, 3h...)
            future_time = now + timedelta(hours=h)
            db_time = future_time.strftime('%Y-%m-%d %H:%M:%S')

            # 3. Check if this specific hour already exists
            cur.execute("SELECT id FROM predictions WHERE location_id = ? AND timestamp = ?", (loc_id, db_time))
            if not cur.fetchone():
                pred_temp, pred_weather, rain_chance = get_forecast_by_name(name, target_time=future_time)

                if pred_temp is not None:
                    cur.execute("""
                        INSERT INTO predictions (location_id, label, predicted_value, predicted_weather_code, rain_chance, timestamp)
                        VALUES (?, 'forecast', ?, ?, ?, ?)
                    """, (loc_id, pred_temp, pred_weather, rain_chance, db_time))

        conn.commit()


def sync_all_locations():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name, lat, lon FROM locations")
    cities = cur.fetchall()

    timestamp = datetime.now().strftime('%Y-%m-%d %H:00:00')
    print(f"\n[*] Syncing {len(cities)} cities at {datetime.now().strftime('%H:%M:%S')}...")

    for city_id, name, lat, lon in cities:
        try:
            cur.execute("SELECT id FROM weather_data WHERE location_id = ? AND timestamp = ?", (city_id, timestamp))
            if cur.fetchone():
                print(f"  [-] {name} already synced. Skipping.")
                continue

            w_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&hourly=relative_humidity_2m,surface_pressure,windspeed_10m,weathercode&timezone=auto"
            a_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=us_aqi"

            w_res = requests.get(w_url).json()
            a_res = requests.get(a_url).json()

            current_w = w_res.get('current_weather', {})
            aqi_val = a_res.get('current', {}).get('us_aqi', 0)

            cur.execute("""
                INSERT INTO weather_data (location_id, timestamp, temperature, humidity, pressure, wind_speed, weather_code, aqi)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (city_id, timestamp, current_w['temperature'], w_res['hourly']['relative_humidity_2m'][0],
                  w_res['hourly']['surface_pressure'][0], current_w['windspeed'], current_w['weathercode'], aqi_val))

            pred_temp, pred_weather, rain_chance = get_forecast_by_name(name)

            cur.execute("""
                INSERT INTO predictions (location_id, label, predicted_value, predicted_weather_code, rain_chance, timestamp)
                VALUES (?, 'forecast', ?, ?, ?, ?)
            """, (city_id, pred_temp, pred_weather, rain_chance, timestamp))

            conn.commit()
            print(f"  [+] {name} synced.")

        except Exception as e:
            print(f"  [!] Error syncing {name}: {e}")
    conn.close()


if __name__ == "__main__":
    load_ai_model()
    db_conn = get_connection()
    try:
        heal_missing_data(db_conn)
    finally:
        db_conn.close()
    sync_all_locations()

    while True:
        sync_all_locations()  # 1. Get the data
        create_dashboard_map()  # 2. Update the map file immediately

        now = datetime.now()
        next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        wait_seconds = (next_hour - now).total_seconds()
        print(f"😴 Sleeping until {next_hour.strftime('%H:%M:%S')}")
        time.sleep(wait_seconds)

