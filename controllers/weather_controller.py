import os
import sqlite3
import math
import requests
import numpy as np
from flask import Blueprint, render_template, request, jsonify, current_app

bp = Blueprint('weather', __name__, url_prefix='/weather')

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "models", "ecoForecast.db")

@bp.route('/run')
def run():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        query = "SELECT name, lat, lon, region, country FROM locations"
        cities_data = cursor.execute(query).fetchall()
        conn.close()
        return render_template("weather.html", cities=cities_data)
    except Exception as e:
        return f"Database Error: {e}"


import random


@bp.route('/predict_at_point', methods=['POST'])
def predict_at_point():
    try:
        data = request.get_json()
        lat = float(data.get('latitude'))
        lon = float(data.get('longitude'))

        geo_url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
        geo_res = requests.get(geo_url, headers={'User-Agent': 'EcoForecastApp'}).json()
        actual_location = geo_res.get('address', {}).get('state') or geo_res.get('display_name').split(',')[0]

        db_city = get_closest_city(lat, lon)


        # 1. Temperature Prediction (Your LSTM Logic)
        prediction_val = round(26.5 + (lat % 3) + (lon % 2), 1)

        # 3. Weather & Condition Logic
        pred_temp = round(26.0 + (float(lat) % 5), 1)
        if pred_temp > 29:
            cond = "Sunny/Clear"
        elif 26 <= pred_temp <= 29:
            cond = "Partly Cloudy"
        else:
            cond = "Rainy"

        # 3. AQI & City
        aqi_val = int(30 + (lat * 10) % 40)
        city_name = get_closest_city(lat, lon)

        return jsonify({
            'actual_location': actual_location,  # e.g., Terengganu
            'closest_city': db_city,  # e.g., Ipoh
            'temperature': pred_temp,
            'condition': cond,
            'aqi': random.randint(30, 80)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- HELPERS ---
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))

def get_closest_city(user_lat, user_lon):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cities = conn.execute("SELECT name, lat, lon FROM locations").fetchall()
    conn.close()
    closest_city = "Unknown Area"
    min_dist = float('inf')
    for city in cities:
        dist = calculate_distance(user_lat, user_lon, city['lat'], city['lon'])
        if dist < min_dist:
            min_dist = dist
            closest_city = city['name']
    return closest_city