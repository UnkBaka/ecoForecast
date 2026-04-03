import sqlite3
import os
from flask import g

DB_PATH = os.path.join(os.path.dirname(__file__), "ecoForecast.db")


def get_connection():
    """Standard connection for scripts (seeding, migration, etc.)."""
    # Added timeout=20 to prevent 'database is locked' errors
    # while the AI background thread is fetching history!
    conn = sqlite3.connect(DB_PATH, timeout=20.0)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def get_db():
    """Flask-friendly connection stored in the 'g' object."""
    if 'db' not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON;")
    return g.db


def close_db(e=None):
    """Closes the database at the end of a request."""
    db = g.pop('db', None)
    if db is not None:
        db.close()


def init_db():
    """Initialises the database schema (safe to call multiple times)."""
    conn = get_connection()
    cur  = conn.cursor()

    # 1. users
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            username   TEXT    UNIQUE NOT NULL,
            password   TEXT    NOT NULL,
            role       TEXT    DEFAULT 'user',
            created_at TEXT    DEFAULT (datetime('now'))
        )
    """)

    # 2. locations
    cur.execute("""
        CREATE TABLE IF NOT EXISTS locations (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            name    TEXT UNIQUE NOT NULL,
            lat     REAL,
            lon     REAL,
            region  TEXT,
            country TEXT DEFAULT 'Malaysia'
        )
    """)

    # 3. weather_data  — includes all columns used by aqi_controller
    cur.execute("""
        CREATE TABLE IF NOT EXISTS weather_data (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            location_id INTEGER NOT NULL,
            timestamp   TEXT    NOT NULL,
            temperature REAL,
            humidity    REAL,
            wind_speed  REAL,
            precipitation REAL,
            pressure    REAL,
            aqi         INTEGER,
            weather_code INTEGER,
            condition   TEXT,
            advice_status TEXT,
            advice_health TEXT,
            advice_reason TEXT,
            FOREIGN KEY (location_id) REFERENCES locations(id)
        )
    """)

    # 4. aqi_data
    cur.execute("""
        CREATE TABLE IF NOT EXISTS aqi_data (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            location_id INTEGER NOT NULL,
            timestamp   TEXT    NOT NULL,
            aqi         INTEGER,
            pm25        REAL,
            pm10        REAL,
            FOREIGN KEY (location_id) REFERENCES locations(id)
        )
    """)

    # 5. weather_history
    cur.execute("""
        CREATE TABLE IF NOT EXISTS weather_history (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            location_name TEXT NOT NULL,
            timestamp     TEXT NOT NULL,
            temperature   REAL,
            condition     TEXT,
            aqi           INTEGER
        )
    """)

    # 6. models
    cur.execute("""
        CREATE TABLE IF NOT EXISTS models (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            version    TEXT,
            algorithm  TEXT,
            trained_on TEXT,
            accuracy   REAL,
            file_path  TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    # 7. predictions  — rain_chance and predicted_weather_code included from the start
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id                     INTEGER PRIMARY KEY AUTOINCREMENT,
            location_id            INTEGER NOT NULL,
            model_id               INTEGER,
            label                  TEXT,
            predicted_value        REAL,
            predicted_weather_code INTEGER,
            rain_chance            REAL DEFAULT 0.0,
            timestamp              TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (location_id) REFERENCES locations(id),
            FOREIGN KEY (model_id)    REFERENCES models(id)
        )
    """)

    # 8. feedback
    cur.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       INTEGER,
            prediction_id INTEGER,
            rating        INTEGER CHECK (rating BETWEEN 1 AND 5),
            comment       TEXT,
            created_at    TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id)       REFERENCES users(id),
            FOREIGN KEY (prediction_id) REFERENCES predictions(id)
        )
    """)

    # 9. logs
    cur.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            action    TEXT,
            timestamp TEXT DEFAULT (datetime('now')),
            details   TEXT
        )
    """)

    # -----------------------------------------------------------------------
    # Safe ALTER TABLE helper — adds a column only when it does not exist yet.
    # Use this instead of bare ALTER TABLE so init_db() stays re-entrant.
    # -----------------------------------------------------------------------
    def safe_add_column(table, col, col_type):
        cur.execute(f"PRAGMA table_info({table})")
        existing = {row[1] for row in cur.fetchall()}
        if col not in existing:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
            print(f"  ✅ Migrated: added {table}.{col}")

    # Back-fill any columns missing from databases created by the old init_db
    safe_add_column("weather_data", "aqi",            "INTEGER")
    safe_add_column("weather_data", "weather_code",   "INTEGER")
    safe_add_column("weather_data", "condition",      "TEXT")
    safe_add_column("weather_data", "advice_status",  "TEXT")
    safe_add_column("weather_data", "advice_health",  "TEXT")
    safe_add_column("weather_data", "advice_reason",  "TEXT")
    safe_add_column("predictions",  "rain_chance",            "REAL DEFAULT 0.0")
    safe_add_column("predictions",  "predicted_weather_code", "INTEGER")

    conn.commit()
    conn.close()
    print("✅ ecoForecast database setup complete!")


if __name__ == "__main__":
    init_db()