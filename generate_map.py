import sqlite3
import pandas as pd
import folium
from folium.plugins import HeatMap
from datetime import datetime
import os

# --- ABSOLUTE PATH LOGIC ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Find the 'EcoForecast' root folder
PROJECT_ROOT = SCRIPT_DIR
while os.path.basename(PROJECT_ROOT) != "EcoForecast" and len(PROJECT_ROOT) > 3:
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

DB_PATH = os.path.join(PROJECT_ROOT, "models", "ecoForecast.db")


def create_dashboard_map():
    if not os.path.exists(DB_PATH):
        print(f"❌ Map Error: Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)

    # Get the latest data
    query = """
    SELECT l.name, l.lat, l.lon, w.temperature, w.aqi, w.humidity
    FROM locations l
    JOIN weather_data w ON l.id = w.location_id
    WHERE w.timestamp = (SELECT MAX(timestamp) FROM weather_data)
    """
    df = pd.read_sql(query, conn)
    conn.close()

    if df.empty:
        print("❌ Map Error: No data found in database to map.")
        return

    # --- CALCULATIONS ---
    df['energy_index'] = df.apply(lambda x: max(0, (x['temperature'] - 24) * (x['humidity'] / 50)), axis=1)
    df['carbon_index'] = df['aqi'] * 0.8

    # Initialize Map
    m = folium.Map(location=[4.2105, 101.9758], zoom_start=6, tiles="cartodbpositron")

    # Layer 1: Temperature Heatmap
    temp_layer = folium.FeatureGroup(name="🔥 Temperature Heatmap", show=True)
    temp_data = df[['lat', 'lon', 'temperature']].values.tolist()
    HeatMap(temp_data, radius=25, blur=15).add_to(temp_layer)
    temp_layer.add_to(m)

    # Layer 2: Energy Footprint (Orange Circles)
    energy_layer = folium.FeatureGroup(name="⚡ Energy Demand", show=False)
    for _, city in df.iterrows():
        folium.CircleMarker(
            location=[city['lat'], city['lon']],
            radius=min(30, city['energy_index'] * 2),  # Cap radius so it doesn't cover the whole screen
            popup=f"<b>{city['name']}</b><br>Cooling Demand: {round(city['energy_index'], 2)} unit",
            color="orange",
            fill=True,
            fill_opacity=0.6
        ).add_to(energy_layer)
    energy_layer.add_to(m)

    # Layer 3: Carbon Footprint (Green/Red Circles)
    carbon_layer = folium.FeatureGroup(name="🌿 Carbon Footprint", show=False)
    for _, city in df.iterrows():
        color = "green" if city['aqi'] < 50 else "red"
        folium.CircleMarker(
            location=[city['lat'], city['lon']],
            radius=min(25, city['carbon_index'] / 2),
            popup=f"<b>{city['name']}</b><br>AQI: {city['aqi']}<br>Carbon Est: {round(city['carbon_index'], 2)}",
            color=color,
            fill=True
        ).add_to(carbon_layer)
    carbon_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # --- SAVE TO VIEWS FOLDER ---
    # This ensures app.py can see it in the templates/views area
    views_folder = os.path.join(PROJECT_ROOT, "views")
    if not os.path.exists(views_folder):
        os.makedirs(views_folder)

    map_path = os.path.join(views_folder, "eco_map.html")
    m.save(map_path)
    print(f"🌍 Map updated at: {datetime.now().strftime('%H:%M:%S')} -> {map_path}")