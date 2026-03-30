from __future__ import annotations  # enables modern union syntax on Python 3.9

import math
import requests
from typing import Optional
from flask import Blueprint, render_template, request
from models.database import get_connection
from datetime import datetime

bp = Blueprint('aqi', __name__, url_prefix='/aqi')

# ---------------------------------------------------------------------------
# Malaysia DOE APIMS reference stations
# Used to show the nearest real monitoring station alongside CAMS model data.
# ---------------------------------------------------------------------------
_DOE_STATIONS = [
    {"name": "Kangar",              "state": "Perlis",          "lat": 6.4449,  "lon": 100.1986},
    {"name": "Alor Setar",          "state": "Kedah",           "lat": 6.1248,  "lon": 100.3673},
    {"name": "Kulim",               "state": "Kedah",           "lat": 5.3656,  "lon": 100.5614},
    {"name": "Langkawi",            "state": "Kedah",           "lat": 6.3500,  "lon":  99.8000},
    {"name": "USM Penang",          "state": "Penang",          "lat": 5.3562,  "lon": 100.3001},
    {"name": "Seberang Perai",      "state": "Penang",          "lat": 5.3983,  "lon": 100.3986},
    {"name": "Balik Pulau",         "state": "Penang",          "lat": 5.3500,  "lon": 100.2333},
    {"name": "Ipoh",                "state": "Perak",           "lat": 4.5975,  "lon": 101.0901},
    {"name": "Taiping",             "state": "Perak",           "lat": 4.8500,  "lon": 100.7333},
    {"name": "Teluk Intan",         "state": "Perak",           "lat": 4.0236,  "lon": 101.0228},
    {"name": "Manjung",             "state": "Perak",           "lat": 4.2167,  "lon": 100.6500},
    {"name": "Petaling Jaya",       "state": "Selangor",        "lat": 3.1073,  "lon": 101.6067},
    {"name": "Shah Alam",           "state": "Selangor",        "lat": 3.0738,  "lon": 101.5183},
    {"name": "Klang",               "state": "Selangor",        "lat": 3.0449,  "lon": 101.4458},
    {"name": "Putrajaya",           "state": "Selangor",        "lat": 2.9264,  "lon": 101.6964},
    {"name": "Cheras",              "state": "Kuala Lumpur",    "lat": 3.0800,  "lon": 101.7333},
    {"name": "Batu Muda KL",        "state": "Kuala Lumpur",    "lat": 3.2100,  "lon": 101.6800},
    {"name": "Nilai",               "state": "Negeri Sembilan", "lat": 2.8167,  "lon": 101.8000},
    {"name": "Port Dickson",        "state": "Negeri Sembilan", "lat": 2.5236,  "lon": 101.7958},
    {"name": "Bandaraya Melaka",    "state": "Melaka",          "lat": 2.1896,  "lon": 102.2501},
    {"name": "Bukit Rambai",        "state": "Melaka",          "lat": 2.2667,  "lon": 102.1833},
    {"name": "Johor Bahru",         "state": "Johor",           "lat": 1.4927,  "lon": 103.7414},
    {"name": "Muar",                "state": "Johor",           "lat": 2.0442,  "lon": 102.5689},
    {"name": "Pasir Gudang",        "state": "Johor",           "lat": 1.4731,  "lon": 103.8972},
    {"name": "Kluang",              "state": "Johor",           "lat": 2.0333,  "lon": 103.3167},
    {"name": "Kuantan",             "state": "Pahang",          "lat": 3.8077,  "lon": 103.3260},
    {"name": "Cameron Highlands",   "state": "Pahang",          "lat": 4.4667,  "lon": 101.3833},
    {"name": "Jerantut",            "state": "Pahang",          "lat": 3.9333,  "lon": 102.3667},
    {"name": "Kuala Terengganu",    "state": "Terengganu",      "lat": 5.3302,  "lon": 103.1408},
    {"name": "Kemaman",             "state": "Terengganu",      "lat": 4.2333,  "lon": 103.4167},
    {"name": "Kota Bharu",          "state": "Kelantan",        "lat": 6.1254,  "lon": 102.2381},
    {"name": "Tanah Merah",         "state": "Kelantan",        "lat": 5.8167,  "lon": 102.1500},
    {"name": "Kota Kinabalu",       "state": "Sabah",           "lat": 5.9804,  "lon": 116.0735},
    {"name": "Sandakan",            "state": "Sabah",           "lat": 5.8402,  "lon": 118.1179},
    {"name": "Tawau",               "state": "Sabah",           "lat": 4.2449,  "lon": 117.8910},
    {"name": "Lahad Datu",          "state": "Sabah",           "lat": 5.0271,  "lon": 118.3300},
    {"name": "Kuching",             "state": "Sarawak",         "lat": 1.5497,  "lon": 110.3592},
    {"name": "Miri",                "state": "Sarawak",         "lat": 4.3995,  "lon": 113.9914},
    {"name": "Sibu",                "state": "Sarawak",         "lat": 2.2866,  "lon": 111.8325},
    {"name": "Bintulu",             "state": "Sarawak",         "lat": 3.1667,  "lon": 113.0333},
]


def get_nearest_doe_station(lat: float, lon: float) -> dict:
    """Return the nearest Malaysia DOE APIMS station to the given coordinates."""
    return min(
        _DOE_STATIONS,
        key=lambda s: math.sqrt((s["lat"] - lat) ** 2 + (s["lon"] - lon) ** 2)
    )


# ---------------------------------------------------------------------------
# Advice engine — returns plain text only (no HTML).
# ---------------------------------------------------------------------------

def generate_smart_advice(aqi: float, temp: float, wind: float) -> dict:
    aqi  = float(aqi  or 0)
    temp = float(temp or 0)
    wind = float(wind or 0)

    if aqi <= 50:
        status = "Good"
        health = "Air quality is ideal for outdoor activities."
    elif aqi <= 100:
        status = "Moderate"
        health = "Sensitive individuals should limit prolonged outdoor exertion."
    elif aqi <= 150:
        status = "Unhealthy"
        health = (
            "Everyone may begin to experience health effects; "
            "members of sensitive groups may experience more serious effects."
        )
    elif aqi <= 200:
        status = "Very Unhealthy"
        health = (
            "Health warnings of emergency conditions. "
            "The entire population is more likely to be affected."
        )
    else:
        status = "Hazardous"
        health = "Health alert: everyone may experience more serious health effects. Avoid all outdoor exertion."

    if aqi > 100:
        if wind < 5:
            reason = (
                "Wind speeds are very low (under 5 km/h), "
                "causing pollutants to stagnate and accumulate in the area."
            )
        elif temp > 32:
            reason = "High temperatures are accelerating smog formation. Stay hydrated."
        else:
            reason = (
                "Regional haze or heavy traffic density is likely "
                "contributing to the elevated index."
            )
    elif aqi <= 50:
        reason = "Strong winds are effectively dispersing pollutants today." if wind > 15 else "Atmospheric conditions are stable and clear."
    else:
        reason = ""

    return {"status": status, "health": health, "reason": reason}


# ---------------------------------------------------------------------------
# External API fetch
# ---------------------------------------------------------------------------

def fetch_external_data(lat: float, lon: float) -> Optional[dict]:
    try:
        w_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current_weather=true"
            f"&hourly=relative_humidity_2m,surface_pressure"
            f"&timezone=auto"
        )
        a_url = (
            f"https://air-quality-api.open-meteo.com/v1/air-quality"
            f"?latitude={lat}&longitude={lon}"
            f"&current=us_aqi,pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,ozone"
            f"&timezone=auto"
        )

        w_res = requests.get(w_url, timeout=10).json()
        a_res = requests.get(a_url, timeout=10).json()

        cw = w_res.get("current_weather", {})
        ca = a_res.get("current", {})

        temp = float(cw.get("temperature") or 0)
        wind = float(cw.get("windspeed")   or 0)
        aqi  = int(ca.get("us_aqi")        or 0)

        def p_level(val, good, mod):
            if val <= good: return "Good"
            if val <= mod:  return "Moderate"
            return "Unhealthy"

        pm25  = round(float(ca.get("pm2_5")            or 0), 1)
        pm10  = round(float(ca.get("pm10")             or 0), 1)
        co    = round(float(ca.get("carbon_monoxide")  or 0) / 1000, 2)
        no2   = round(float(ca.get("nitrogen_dioxide") or 0), 1)
        ozone = round(float(ca.get("ozone")            or 0), 1)

        pollutants = [
            {"name": "PM2.5", "icon": "🌫️", "value": pm25,  "unit": "µg/m³", "pct": min(pm25  / 75  * 100, 100), "level": p_level(pm25,  12,   35)},
            {"name": "PM10",  "icon": "💨",  "value": pm10,  "unit": "µg/m³", "pct": min(pm10  / 154 * 100, 100), "level": p_level(pm10,  54,  154)},
            {"name": "CO",    "icon": "🏭",  "value": co,    "unit": "mg/m³", "pct": min(co    / 15  * 100, 100), "level": p_level(co,   4.4,  9.4)},
            {"name": "NO2",   "icon": "🔴",  "value": no2,   "unit": "µg/m³", "pct": min(no2   / 200 * 100, 100), "level": p_level(no2,  53,  100)},
            {"name": "Ozone", "icon": "sunny",  "value": ozone, "unit": "µg/m³", "pct": min(ozone / 180 * 100, 100), "level": p_level(ozone, 54,  70)},
        ]

        return {
            "temp":     temp,
            "wind":     wind,
            "humidity": float((w_res.get("hourly", {}).get("relative_humidity_2m") or [0])[0]),
            "pressure": float((w_res.get("hourly", {}).get("surface_pressure")     or [0])[0]),
            "aqi":      aqi,
            "code":     int(cw.get("weathercode") or 0),
            "pollutants": pollutants,
        }

    except Exception as e:
        print(f"[aqi] Error fetching external data: {e}")
        return None


# ---------------------------------------------------------------------------
# History query — subquery pulls the latest advice text per hour so it is
# never NULLed out by the AVG() GROUP BY aggregation.
# ---------------------------------------------------------------------------

def get_city_history(conn, city_name: str) -> list:
    if not city_name:
        return []

    cur = conn.cursor()
    query = """
        SELECT
            strftime('%Y-%m-%d %H:00', w.timestamp) AS hour,
            ROUND(AVG(w.aqi), 0)                    AS aqi,
            ROUND(AVG(w.temperature), 1)             AS temp,
            (
                SELECT w2.advice_status
                FROM weather_data w2
                WHERE w2.location_id = w.location_id
                  AND strftime('%Y-%m-%d %H', w2.timestamp)
                      = strftime('%Y-%m-%d %H', w.timestamp)
                ORDER BY w2.timestamp DESC
                LIMIT 1
            ) AS advice_status,
            (
                SELECT w2.advice_health
                FROM weather_data w2
                WHERE w2.location_id = w.location_id
                  AND strftime('%Y-%m-%d %H', w2.timestamp)
                      = strftime('%Y-%m-%d %H', w.timestamp)
                ORDER BY w2.timestamp DESC
                LIMIT 1
            ) AS advice_health,
            (
                SELECT w2.advice_reason
                FROM weather_data w2
                WHERE w2.location_id = w.location_id
                  AND strftime('%Y-%m-%d %H', w2.timestamp)
                      = strftime('%Y-%m-%d %H', w.timestamp)
                ORDER BY w2.timestamp DESC
                LIMIT 1
            ) AS advice_reason
        FROM weather_data w
        JOIN locations l ON w.location_id = l.id
        WHERE w.aqi IS NOT NULL
          AND w.aqi > 0
          AND l.name LIKE ?
        GROUP BY strftime('%Y-%m-%d %H', w.timestamp)
        ORDER BY hour DESC
        LIMIT 24
    """

    safe_name = city_name.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")
    rows = cur.execute(query, (f"%{safe_name}%",)).fetchall()

    result = []
    for r in rows:
        aqi_val  = float(r["aqi"]  or 0)
        temp_val = float(r["temp"] or 0)
        wind_val = 0.0  # wind not stored per-hour; re-derive advice from aqi+temp

        # If advice columns are NULL (rows inserted before migration),
        # regenerate them on the fly so the table is never empty.
        status = r["advice_status"]
        health = r["advice_health"]
        reason = r["advice_reason"]

        if not status:
            fallback = generate_smart_advice(aqi_val, temp_val, wind_val)
            status = fallback["status"]
            health = fallback["health"]
            reason = fallback["reason"]

        advice_parts = [f"<strong>{status}:</strong> {health}"]
        if reason:
            advice_parts.append(f"<small class='text-muted'>{reason}</small>")

        result.append({
            "time":   r["hour"],
            "aqi":    aqi_val,
            "temp":   temp_val,
            "advice": "<br>".join(advice_parts),
        })

    return result


# ---------------------------------------------------------------------------
# Main route
# ---------------------------------------------------------------------------

@bp.route('/run', methods=["GET", "POST"])
def run():
    import sqlite3
    conn = get_connection()
    conn.row_factory = sqlite3.Row

    selected_city    = None
    current_data     = None
    error_msg        = None
    search_performed = False

    try:
        cur = conn.cursor()

        cur.execute("SELECT name FROM locations ORDER BY name")
        all_cities = [row["name"] for row in cur.fetchall()]

        if request.method == "POST":
            input_city = (request.form.get("city") or "").strip()

            if not input_city:
                error_msg = "Please enter a city name to check the AQI."
            else:
                safe_input = (
                    input_city
                    .replace("\\", "\\\\")
                    .replace("%", r"\%")
                    .replace("_", r"\_")
                )
                cur.execute(
                    "SELECT id, name, lat, lon FROM locations WHERE name LIKE ? ESCAPE '\\'",
                    (f"%{safe_input}%",),
                )
                location = cur.fetchone()

                if location:
                    loc_id        = location["id"]
                    real_name     = location["name"]
                    lat           = location["lat"]
                    lon           = location["lon"]
                    selected_city    = real_name
                    search_performed = True

                    data = fetch_external_data(lat, lon)

                    if data:
                        timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        advice_dict = generate_smart_advice(data["aqi"], data["temp"], data["wind"])

                        cur.execute(
                            """
                            SELECT id FROM weather_data
                            WHERE location_id = ?
                              AND timestamp >= datetime('now', '-30 minutes')
                            """,
                            (loc_id,),
                        )
                        if not cur.fetchone():
                            cur.execute(
                                """
                                INSERT INTO weather_data (
                                    location_id, timestamp,
                                    temperature, humidity, pressure,
                                    wind_speed, weather_code, aqi,
                                    advice_status, advice_health, advice_reason
                                )
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    loc_id, timestamp,
                                    data["temp"], data["humidity"], data["pressure"],
                                    data["wind"], data["code"],    data["aqi"],
                                    advice_dict["status"],
                                    advice_dict["health"],
                                    advice_dict["reason"],
                                ),
                            )
                            conn.commit()

                        station = get_nearest_doe_station(lat, lon)
                        current_data = {
                            **data,
                            "advice":       advice_dict,
                            "time":         timestamp,
                            "doe_station":  station["name"],
                            "doe_state":    station["state"],
                        }
                else:
                    error_msg = f"City '{input_city}' not found in our Malaysian database."

        history = get_city_history(conn, selected_city)

    finally:
        conn.close()

    return render_template(
        "aqi.html",
        aqi_data=current_data,
        city=selected_city,
        history=history,
        all_cities=all_cities,
        error_msg=error_msg,
        show_aqi_section=search_performed,
    )