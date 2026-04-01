import os
import json
import sqlite3
import pandas as pd
import requests
from groq import Groq  # NEW IMPORT
from flask import Flask, render_template, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

# Import DB helpers
from models.database import close_db, get_connection

# Import Blueprints
from controllers.aqi_controller import bp as aqi_bp
from controllers.weather_controller import bp as weather_bp
# Import the AI Service
from prediction_service import get_forecast_by_name

# --- Weather code from Meteo website ---
WEATHER_DESCRIPTIONS = {
    0: "Clear sky ☀️",
    1: "Mainly clear 🌤️",
    2: "Partly cloudy ⛅",
    3: "Overcast ☁️",
    45: "Fog 🌫️",
    48: "Depositing rime fog 🌫️",
    51: "Light drizzle 🌧️",
    53: "Moderate drizzle 🌧️",
    55: "Dense drizzle 🌧️",
    56: "Light freezing drizzle ❄️",
    57: "Dense freezing drizzle ❄️",
    61: "Slight rain 🌧️",
    63: "Moderate rain 🌧️",
    65: "Heavy rain 🌧️",
    66: "Light freezing rain ❄️🌧️",
    67: "Heavy freezing rain ❄️🌧️",
    71: "Slight snow fall ❄️",
    73: "Moderate snow fall ❄️",
    75: "Heavy snow fall ❄️",
    77: "Snow grains ❄️",
    80: "Slight rain showers 🌦️",
    81: "Moderate rain showers 🌦️",
    82: "Violent rain showers ⛈️",
    85: "Slight snow showers ❄️",
    86: "Heavy snow showers ❄️",
    95: "Thunderstorm ⛈️",
    96: "Thunderstorm with slight hail ⛈️🧊",
    99: "Thunderstorm with heavy hail ⛈️🧊"
}

client = Groq(api_key="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")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "models", "ecoForecast.db")
print(f"✅ Database connected at: {DB_PATH}")

CITY_FILE = 'cities.json'

if not os.path.exists(CITY_FILE):
    default_cities = [
        {"name": "Kuala Lumpur", "lat": 3.1478, "lng": 101.6953},
        {"name": "George Town", "lat": 5.4144, "lng": 100.3292}
    ]
    with open(CITY_FILE, 'w') as f:
        json.dump(default_cities, f)


def create_app():
    # Note: Based on your uploaded files, your HTML is in 'templates', not 'views'
    app = Flask(__name__, template_folder="views", static_folder="static")

    # Close database connection after each request
    app.teardown_appcontext(close_db)

    # --- MODEL LOADING ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "weather_lstm_model.keras")

    try:
        app.my_lstm_model = load_model(MODEL_PATH)
        print(f" Model loaded successfully")
    except Exception as e:
        app.my_lstm_model = None
        print(f" Failed to load model: {e}")

    # Register Blueprints
    app.register_blueprint(aqi_bp)
    app.register_blueprint(weather_bp)

    # --- ROUTES ---
    @app.route('/')
    def index():
        return render_template('landing.html')

    @app.route('/footprint')
    def footprint():
        return render_template('footprint.html')

    @app.route('/admin')
    def admin_page():
        key = request.args.get('key')
        if key != "eco2026":
            return "<h1>403 Forbidden</h1><p>Access Denied. Contact System Admin.</p>", 403
        # Fetch current cities so the admin can see what's already there
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT name, lat, lon FROM locations")
        cities = cur.fetchall()
        conn.close()

        return render_template('admin.html', cities=cities)

    @app.route('/admin/add_city', methods=['POST'])
    def add_new_city():
        try:
            data = request.json
            name = data.get('name')
            lat = float(data.get('lat'))
            lng = float(data.get('lng'))

            conn = get_connection()
            cur = conn.cursor()
            # INSERT OR IGNORE prevents duplicates if you click the same spot twice
            cur.execute("INSERT OR IGNORE INTO locations (name, lat, lon) VALUES (?, ?, ?)", (name, lat, lng))
            conn.commit()
            conn.close()

            print(f" Admin Added City: {name} ({lat}, {lng})")
            return jsonify({"status": "success", "message": f"{name} added to database!"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    from flask import jsonify

    @app.route('/get_heatmap_data')
    def get_heatmap_data():
        try:
            conn = sqlite3.connect(DB_PATH)
            query = """
            SELECT l.lat, l.lon, l.name, w.temperature, w.humidity
            FROM locations l
            JOIN weather_data w ON l.id = w.location_id
            WHERE w.timestamp = (SELECT MAX(timestamp) FROM weather_data)
            """
            df = pd.read_sql(query, conn)
            conn.close()

            if df.empty:
                return jsonify([])

            heatmap_points = []
            for _, row in df.iterrows():
                temp = row['temperature']
                humidity = row['humidity']
                name = row['name']

                # Heat advisory logic
                if temp >= 37:
                    advisory = "🚨 Extreme Heat! Stay indoors."
                elif temp >= 34:
                    advisory = "🔴 Very Hot! Seek shade immediately."
                elif temp >= 31:
                    advisory = "🟠 Hot & Humid. Limit outdoor activity."
                elif temp >= 28:
                    advisory = "🟡 Warm. Stay hydrated."
                else:
                    advisory = "🟢 Pleasant. Comfortable outdoors."

                heatmap_points.append([
                    row['lat'],
                    row['lon'],
                    temp,  # index 2
                    humidity,  # index 3
                    name,  # index 4
                    advisory  # index 5
                ])

            return jsonify(heatmap_points)

        except Exception as e:
            print(f"❌ Database Error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/admin/delete_city/<name>', methods=['POST'])
    def delete_city(name):
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM locations WHERE name = ?", (name,))
        conn.commit()
        conn.close()
        return jsonify({"message": f"{name} removed."})

    @app.route('/analysis')
    def analysis_page():
        return render_template('results.html')

    @app.route('/get_locations')
    def get_locations():
        conn = sqlite3.connect(DB_PATH)
        # We only need id, name, lat, and lon for the pins
        query = "SELECT id, name, lat, lon FROM locations"
        df = pd.read_sql(query, conn)
        conn.close()
        return jsonify(df.to_dict(orient='records'))

    @app.route('/get_history')
    def get_history():
        location = request.args.get('location', '')
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute("""
                SELECT DATE(w.timestamp) as date, 
                       ROUND(AVG(w.temperature), 1) as temp,
                       ROUND(AVG(w.aqi), 0) as aqi
                FROM weather_data w
                JOIN locations l ON w.location_id = l.id
                WHERE l.name LIKE ?
                GROUP BY DATE(w.timestamp)
                ORDER BY date DESC
                LIMIT 10
            """, (f"%{location}%",))
            rows = cur.fetchall()
            conn.close()
            return jsonify([{"date": r[0], "temp": r[1], "aqi": r[2] or 0} for r in rows])
        except Exception as e:
            print(f"❌ get_history error: {e}")
            return jsonify([]), 500

    @app.route('/results')
    def results_page():
        conn = get_connection()
        try:
            cur = conn.cursor()

            # 1. Fetch the exact list of all cities (Should be 10)
            cities = [row[0] for row in cur.execute("SELECT name FROM locations ORDER BY name ASC").fetchall()]

            # 2. Fetch the actual database predictions
            query = """
                SELECT 
                    l.name as city,
                    p.predicted_value,          
                    w.temperature as actual_value, 
                    p.timestamp as pred_time,   
                    w.weather_code,             
                    p.predicted_weather_code,
                    p.rain_chance    
                FROM predictions p
                JOIN locations l ON p.location_id = l.id
                LEFT JOIN weather_data w ON p.location_id = w.location_id 
                                        AND w.timestamp = p.timestamp
                WHERE p.label = 'forecast'
            """
            raw_data = cur.execute(query).fetchall()

            if not raw_data:
                return render_template('results.html', weather=[])

            # 3. Organize the raw data into a dictionary grouped by (Hour, City)
            record_dict = {}
            min_time = None

            for row in raw_data:
                city = row[0]
                # Convert DB string to a Python datetime object safely
                try:
                    dt = datetime.strptime(row[3][:19], '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    continue

                # Round down to the top of the hour (e.g., 16:28 -> 16:00)
                hour_dt = dt.replace(minute=0, second=0, microsecond=0)

                # Find the oldest record to know where to start our timeline
                if min_time is None or hour_dt < min_time:
                    min_time = hour_dt

                # Save to dictionary
                record_dict[(hour_dt, city)] = {
                    'ai_temp': row[1],
                    'act_temp': row[2],
                    'act_weather': row[4],
                    'ai_weather': row[5],
                    'rain_chance': row[6]
                }

            # 4. Determine the timeline boundaries
            if min_time is None:
                min_time = datetime.now().replace(minute=0, second=0, microsecond=0)

            max_time = datetime.now().replace(minute=0, second=0, microsecond=0)

            # Safety catch for timezone differences
            if max_time < min_time:
                max_time = min_time

            # 5. Generate the perfect 10-city hourly grid (Newest to Oldest)
            final_data = []
            current_time = max_time

            while current_time >= min_time:
                time_str = current_time.strftime('%Y-%m-%d %H:00')

                for city in cities:
                    key = (current_time, city)
                    if key in record_dict:
                        data = record_dict[key]
                        raw_act_code = data['act_weather']
                        raw_ai_code = data['ai_weather']

                        act_text = WEATHER_DESCRIPTIONS.get(raw_act_code,
                                                            "Unknown") if raw_act_code is not None else None
                        ai_text = WEATHER_DESCRIPTIONS.get(raw_ai_code, "Unknown") if raw_ai_code is not None else None

                        final_data.append((
                            city,
                            data['ai_temp'],
                            data['act_temp'],
                            time_str,
                            act_text,
                            ai_text,
                            data['rain_chance']
                        ))
                    else:
                        final_data.append((
                            city, None, None, time_str, None, None, None
                        ))

                current_time -= timedelta(hours=1)

            unique_dates = sorted(list(set(row[3][:10] for row in final_data)))

            return render_template('results.html', weather=final_data, available_dates=unique_dates)

        finally:
            conn.close()

    # --- API ROUTES ---
    @app.route('/api/predict', methods=['GET'])
    def predict_weather():
        city = request.args.get('city', 'Kuala Lumpur')
        result = get_forecast_by_name(city)

        if isinstance(result, float):
            return jsonify({
                "status": "success",
                "city": city,
                "prediction": result,
                "unit": "°C"
            })
        else:
            return jsonify({"status": "error", "message": "Prediction failed"}), 400

    @app.route('/api/predict_detailed')
    def predict_detailed():
        city = request.args.get('city', 'Kuala Lumpur')

        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT id FROM locations WHERE name LIKE ?", (f"%{city}%",))
        loc_row = cur.fetchone()

        current_temp = 30.0
        if loc_row:
            cur.execute("SELECT temperature FROM weather_data WHERE location_id=? ORDER BY timestamp DESC LIMIT 1",
                        (loc_row[0],))
            row = cur.fetchone()
            if row: current_temp = row[0]
        conn.close()

        target_temp = get_forecast_by_name(city)
        if not isinstance(target_temp, float):
            target_temp = current_temp

        diff = target_temp - current_temp

        detailed = [
            {"time": "15 mins", "temp": round(current_temp + (diff * 0.25), 1)},
            {"time": "30 mins", "temp": round(current_temp + (diff * 0.50), 1)},
            {"time": "45 mins", "temp": round(current_temp + (diff * 0.75), 1)},
            {"time": "60 mins", "temp": round(target_temp, 1)}
        ]

        return jsonify(detailed)

    @app.route('/api/history/<int:location_id>')
    def graph_data(location_id):
        conn = get_connection()
        try:
            query = """
                    SELECT timestamp, MAX(temperature) as temperature, MAX(aqi) as aqi 
                    FROM weather_data 
                    WHERE location_id = ? 
                    GROUP BY timestamp 
                    ORDER BY timestamp DESC LIMIT 50
                """
            df = pd.read_sql(query, conn, params=(location_id,))

            if df.empty:
                return jsonify({
                    "labels": [], "temp": [], "aqi": [],
                    "stats": {"avg_temp": 0, "max_temp": 0, "min_temp": 0, "max_aqi": 0}
                })

            df = df.iloc[::-1]

            stats_query = """
                    SELECT ROUND(AVG(temperature), 1) as avg_temp, 
                           MAX(temperature) as max_temp, 
                           MIN(temperature) as min_temp, 
                           MAX(aqi) as max_aqi 
                    FROM weather_data 
                    WHERE location_id = ?
                """
            stats_df = pd.read_sql(stats_query, conn, params=(location_id,))

            if not stats_df.empty and stats_df.iloc[0]['max_temp'] is not None:
                stats = stats_df.to_dict('records')[0]
            else:
                stats = {"avg_temp": 0.0, "max_temp": 0.0, "min_temp": 0.0, "max_aqi": 0}

            return jsonify({
                "labels": pd.to_datetime(df['timestamp']).dt.strftime('%H:%M').tolist(),
                "temp": df['temperature'].tolist(),
                "aqi": df['aqi'].fillna(0).tolist(),
                "stats": stats
            })

        except Exception as e:
            print(f"❌ API Error: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            conn.close()

    @app.route('/api/aqi_analysis', methods=['POST'])
    def aqi_analysis():
        try:
            data = request.json
            city = data.get('city', 'this location')
            aqi = float(data.get('aqi', 0))
            temp = float(data.get('temp', 0))
            wind = float(data.get('wind', 0))

            # Determine category label for context
            if aqi <= 50:
                category = "Good"
            elif aqi <= 100:
                category = "Moderate"
            elif aqi <= 150:
                category = "Unhealthy for Sensitive Groups"
            elif aqi <= 200:
                category = "Unhealthy"
            else:
                category = "Very Unhealthy / Hazardous"

            prompt = f"""You are EcoForecast AI, an air quality assistant for Malaysia.

    Current conditions in {city}:
    - AQI: {aqi} ({category})
    - Temperature: {temp}°C
    - Wind Speed: {wind} km/h

    Write a concise environmental analysis in exactly this HTML structure (no markdown, no extra tags):

    <div class="row">
      <div class="col-md-6 border-end pe-md-4 mb-4 mb-md-0">
        <h6 class="text-uppercase text-muted fw-bold mb-2"><i class="bi bi-cloud-haze2 me-1"></i>Atmospheric Conditions</h6>
        <p style="font-size:0.92rem;">[2-3 sentences about current AQI level, what is causing it, and how wind/temperature are affecting pollutant dispersal in {city}. Be specific to Malaysia's climate context.]</p>
      </div>
      <div class="col-md-6 ps-md-4">
        <h6 class="text-uppercase text-muted fw-bold mb-2"><i class="bi bi-lungs me-1"></i>Health &amp; Activity Advisory</h6>
        <ul style="font-size:0.92rem; padding-left:1.2rem;">
          <li><strong>General Public:</strong> [specific advice]</li>
          <li><strong>Sensitive Groups:</strong> [specific advice for elderly, children, asthma patients]</li>
          <li><strong>Best Outdoor Time:</strong> [recommend specific time of day, e.g. early morning 6-8am, based on AQI and conditions]</li>
          <li><strong>Precaution:</strong> [one practical tip e.g. mask type, hydration, indoor ventilation]</li>
        </ul>
      </div>
    </div>

    Rules:
    - Write ONLY the HTML above. No preamble, no explanation, no markdown fences.
    - Keep each point concise (1 sentence max per bullet).
    - Be factual and helpful, not alarmist.
    - Reference {city} and Malaysia context naturally."""

            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system",
                     "content": "You are a precise air quality analyst. Output only clean HTML as instructed. No "
                                "markdown, no extra commentary."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.1-8b-instant",
                max_tokens=600,
                temperature=0.4,
            )

            raw = chat_completion.choices[0].message.content.strip()

            # Strip any accidental markdown fences the model might add
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("html"):
                    raw = raw[4:]
                raw = raw.strip()
            if raw.endswith("```"):
                raw = raw[:-3].strip()

            return jsonify({"html": raw})

        except Exception as e:
            print(f"[aqi_analysis] Error: {e}")
            return jsonify(
                {"html": "<p class='text-danger'>AI analysis temporarily unavailable. Please try again.</p>"}), 500

    @app.route('/api/chat', methods=['POST'])
    def ai_chat():
        try:
            data = request.json
            user_query = data.get('query', '')
            user_lat = data.get('lat')
            user_lng = data.get('lng')
            history = data.get('history', [])

            real_temp = data.get('real_temp')
            real_rain = data.get('real_rain')
            real_aqi = data.get('real_aqi')
            real_condition = data.get('real_condition')
            real_city = data.get('real_city')

            # 1. Reverse-geocode GPS to neighbourhood name
            neighbourhood_label = None
            if user_lat and user_lng:
                try:
                    rev_url = (
                        f"https://nominatim.openstreetmap.org/reverse"
                        f"?lat={user_lat}&lon={user_lng}&format=json&zoom=16&addressdetails=1"
                    )
                    rev_res = requests.get(rev_url, timeout=4, headers={"User-Agent": "EcoForecast/1.0"}).json()
                    addr = rev_res.get("address", {})
                    neighbourhood = (
                            addr.get("neighbourhood") or addr.get("suburb") or
                            addr.get("quarter") or addr.get("village") or
                            addr.get("city_district") or addr.get("town") or
                            addr.get("city") or ""
                    )
                    city_name = addr.get("city") or addr.get("town") or addr.get("county") or ""
                    state_name = addr.get("state") or ""
                    if neighbourhood and city_name and neighbourhood != city_name:
                        neighbourhood_label = f"{neighbourhood}, {city_name}, {state_name}"
                    elif city_name:
                        neighbourhood_label = f"{city_name}, {state_name}"
                except Exception as geo_err:
                    print(f"[chat] Reverse geocode failed: {geo_err}")

            user_place = neighbourhood_label or (f"Lat {user_lat}, Lng {user_lng}" if user_lat else "Unknown location")

            # 2. Fallback — fetch live data for GPS if no clicked-point data
            if real_temp is None and user_lat and user_lng:
                try:
                    w_fb = requests.get(
                        f"https://api.open-meteo.com/v1/forecast?latitude={user_lat}&longitude={user_lng}"
                        f"&current_weather=true&hourly=precipitation_probability&timezone=auto", timeout=5
                    ).json()
                    a_fb = requests.get(
                        f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={user_lat}&longitude={user_lng}"
                        f"&current=us_aqi&timezone=auto", timeout=5
                    ).json()
                    cur_hour = datetime.now().hour
                    precip_fb = w_fb.get("hourly", {}).get("precipitation_probability", [0])
                    real_rain = precip_fb[cur_hour] if cur_hour < len(precip_fb) else 0
                    real_temp = w_fb.get("current_weather", {}).get("temperature")
                    real_condition = w_fb.get("current_weather", {}).get("weathercode", "")
                    real_aqi = a_fb.get("current", {}).get("us_aqi", "Unknown")
                    real_city = user_place
                except Exception as fb_err:
                    print(f"[chat] Fallback weather fetch failed: {fb_err}")

            # 3. Weather code to plain English
            weather_code_map = {
                0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
                45: "Foggy", 48: "Foggy", 51: "Light drizzle", 53: "Drizzle",
                55: "Heavy drizzle", 61: "Light rain", 63: "Moderate rain", 65: "Heavy rain",
                80: "Rain showers", 81: "Rain showers", 82: "Heavy showers",
                95: "Thunderstorm", 96: "Thunderstorm",
            }
            try:
                condition_text = weather_code_map.get(int(float(real_condition or 0)), "Partly cloudy")
            except (ValueError, TypeError):
                condition_text = "Partly cloudy"

            # 4. Build location context
            if real_temp is not None and real_rain is not None:
                current_location_context = (
                    f"USER'S CURRENT LOCATION: {user_place}\n"
                    f"- Nearest monitored city: {real_city or 'Unknown'}\n"
                    f"- Temperature: {real_temp}°C\n"
                    f"- Condition: {condition_text}\n"
                    f"- Rain probability: {real_rain}%\n"
                    f"- AQI: {real_aqi}\n"
                )
            else:
                current_location_context = (
                    f"USER'S LOCATION: {user_place}\n"
                    "No live weather data available yet. Ask the user to click a point on the map."
                )

            # Clicked city context (real data from predict_on_point, not DB)
            clicked_city_name = data.get('clicked_city_name')
            print(
                f"[chat] clicked_city={clicked_city_name} aqi={data.get('clicked_city_aqi')} rain={data.get('clicked_city_rain')}")
            clicked_city_temp = data.get('clicked_city_temp')
            clicked_city_aqi = data.get('clicked_city_aqi')
            clicked_city_rain = data.get('clicked_city_rain')

            clicked_city_context = ""
            if clicked_city_name:
                clicked_city_context = (
                    f"\nLAST CLICKED CITY (real-time data from map pin):\n"
                    f"- City: {clicked_city_name}\n"
                    f"- Temperature: {clicked_city_temp}°C\n"
                    f"- AQI: {clicked_city_aqi}\n"
                    f"- Rain probability: {clicked_city_rain}%\n"
                    f"Use this data when the user asks about {clicked_city_name}.\n"
                )

            # 5. Malaysia city DB context
            conn = get_connection()
            cur = conn.cursor()
            cur.execute("""
                SELECT l.name, w.temperature, w.aqi
                FROM locations l
                LEFT JOIN (
                    SELECT location_id, temperature, aqi
                    FROM weather_data
                    WHERE id IN (SELECT MAX(id) FROM weather_data GROUP BY location_id)
                ) w ON l.id = w.location_id
            """)
            latest_data = cur.fetchall()
            conn.close()

            malaysia_context = "MALAYSIA CITY DATA:\n" + "\n".join(
                [
                    f"- {c}: {round(t, 1)}°C" +
                    (f", AQI {a}" if a and int(a) > 0 else ", AQI not available")
                    for c, t, a in latest_data if t
                ]
            )
            known_cities_str = ", ".join([row[0] for row in latest_data if row[1]])
            # Add a note about Penang alias
            malaysia_context += "\nNote: George Town = Penang island city. Penang/Pinang refers to George Town."

            # 6. System prompt
            system_instruction = f"""
    You are EcoForecast AI, a weather and air quality assistant for Malaysia ONLY.

    {current_location_context}
    {clicked_city_context}
    {malaysia_context}

    STRICT RULES:
    1. LOCATION: Always refer to the user as being in "{user_place}". Never show raw coordinates.
    2. RAIN: Rain probability at user's location is exactly {real_rain}%. Never invent this number.
    3. CITY LOOKUP: First check LAST CLICKED CITY data above — if the user asks about that city, use its exact real-time numbers including rain probability. Then check MALAYSIA CITY DATA for temperature. For cities not in either source, say you don't have data.
    5. OFF-TOPIC: If user asks anything unrelated to weather, air quality, or health impacts — reply ONLY: "I can only help with weather and air quality questions! 🌤️ Try asking about the weather in your area or any Malaysian city."
    6. CONTEXT: If user asks "will it rain" or "how is it" without naming a city, check conversation history for the last mentioned city and answer for that city. If no city was mentioned, answer for user's location: {user_place}.
    7. FORMAT: 1-2 short friendly sentences. Emojis welcome. No bullet points.
    """

            # 7. Build messages with full conversation history
            messages = [{"role": "system", "content": system_instruction}]
            if len(history) > 1:
                messages.extend(history[:-1])  # all except current (already in user_query)
            messages.append({"role": "user", "content": user_query})

            chat_completion = client.chat.completions.create(
                messages=messages,
                model="llama-3.1-8b-instant",
            )

            return jsonify({"response": chat_completion.choices[0].message.content})

        except Exception as e:
            print("AI Error:", e)
            return jsonify({"response": "I'm having trouble connecting to my neural network. Try again!"}), 500

    @app.route('/api/get_city_coords', methods=['GET'])
    def get_city_coords():
        city = request.args.get('city', '').strip()

        # Expand common Malaysian abbreviations
        abbreviations = {
            'kl': 'Kuala Lumpur',
            'jb': 'Johor Bahru',
            'kk': 'Kota Kinabalu',
            'kb': 'Kota Bharu',
            'penang': 'George Town',
            'pinang': 'George Town',
            'sp': 'Sungai Petani',
            'melaka': 'Malacca City',
            'melacca': 'Malacca City',
            'pj': 'Petaling Jaya',
            'ipoh': 'Ipoh',
        }
        city_lookup = abbreviations.get(city.lower(), city)

        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT name, lat, lon FROM locations WHERE name LIKE ? LIMIT 1",
            (f"%{city_lookup}%",)
        )
        row = cur.fetchone()
        conn.close()
        if row:
            return jsonify({"found": True, "name": row[0], "lat": row[1], "lon": row[2]})
        return jsonify({"found": False})

    @app.route('/weather/predict_on_point', methods=['POST'])
    def predict_on_point():
        data = request.json
        lat = data.get('latitude')
        lng = data.get('longitude')
        current_hour = datetime.now().hour

        def get_nearest_city(lat, lng):
            conn = get_connection()
            cur = conn.cursor()

            # Fetch all cities currently in your DB
            cur.execute("SELECT name, lat, lon FROM locations")
            db_cities = cur.fetchall()  # Returns a list of tuples like seed()
            conn.close()

            if not db_cities:
                return "Kuala Lumpur"  # Fallback if DB is empty

            # Use the same logic as before, but on the live DB list
            closest = min(db_cities, key=lambda c: (float(lat) - c[1]) ** 2 + (float(lng) - c[2]) ** 2)
            return closest[0]

        city_name = get_nearest_city(lat, lng)
        try:
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&current_weather=true&hourly=temperature_2m,relative_humidity_2m,weathercode,precipitation_probability&timezone=auto"
            a_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lng}&current=us_aqi"

            w_res = requests.get(url).json()
            a_res = requests.get(a_url).json()

            hourly = w_res.get('hourly', {})
            current_w = w_res.get('current_weather', {})
            time_list = hourly.get('time', [])

            # ✅ Add precipitation_probability to your Open-Meteo URL
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&current_weather=true&hourly=temperature_2m,relative_humidity_2m,weathercode,precipitation_probability&timezone=auto"

            # Then get current hour's rain probability
            precip_list = hourly.get('precipitation_probability', [])
            base_rain_chance = precip_list[current_hour] if current_hour < len(precip_list) else 0.0

            forecast_list = []
            offsets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                       13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                       48, 72, 96, 120, 144, 168]

            for offset in offsets:
                i = current_hour + offset
                if i < len(hourly.get('temperature_2m', [])):
                    temp = hourly['temperature_2m'][i]
                    raw_time = time_list[i]
                    dt_obj = datetime.fromisoformat(raw_time)

                    if offset == 0:
                        display_time = "Now"
                    elif offset < 24:
                        display_time = dt_obj.strftime("%#I %p")  # Windows
                        # display_time = dt_obj.strftime("%-I %p")  # Linux/Mac
                    else:
                        display_time = dt_obj.strftime("%a, %d %b")

                    # Use real precipitation probability, fallback to base
                    p = precip_list[i] if i < len(precip_list) else base_rain_chance

                    forecast_list.append({
                        "time": display_time,
                        "temp": temp,
                        "prob": round(float(p), 1)
                    })

            print(f"DEBUG rain chances: {[f['prob'] for f in forecast_list]}")

            # Summary logic (OUTSIDE the for loop)
            if base_rain_chance > 80:
                summary = "⚠️ High probability of rain. Bring an umbrella!"
            elif base_rain_chance > 40:
                summary = "☁️ Unsettled weather. Might rain later."
            else:
                summary = "☀️ Mostly clear skies expected."

            response_data = {
                "temperature": current_w.get('temperature', 0),
                "condition": WEATHER_DESCRIPTIONS.get(current_w.get('weathercode'), "Cloudy"),
                "aqi": a_res.get('current', {}).get('us_aqi', 0),
                "actual_location": f"Point ({lat}, {lng})",
                "closest_city": city_name,
                "rain_percentage": base_rain_chance,
                "ai_summary": summary,
                "forecast_data": forecast_list
            }

            print(f"DEBUG: Sending to UI -> {city_name} success")
            return jsonify(response_data)

        except Exception as e:
            print(f"❌ CRITICAL ROUTE ERROR: {e}")
            return jsonify({"error": str(e), "rain_percentage": 0, "forecast_data": []}), 500

    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
