"""
Run this script ONCE to clean up duplicate and null records in weather_data.
Safe to run multiple times — it won't delete good data.
"""
import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "models", "ecoForecast.db")

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# 1. Count before
cur.execute("SELECT COUNT(*) FROM weather_data")
before = cur.fetchone()[0]
print(f"📊 Records before cleanup: {before}")

# 2. Delete rows where ALL key fields are null (useless old records)
cur.execute("""
    DELETE FROM weather_data
    WHERE temperature IS NULL
    AND humidity IS NULL
    AND pressure IS NULL
    AND wind_speed IS NULL
    AND condition IS NULL
""")
deleted_null = cur.rowcount
print(f"🗑️  Deleted {deleted_null} empty/null records")

# 3. Delete duplicates — keep only the MIN(id) per location per minute
cur.execute("""
    DELETE FROM weather_data
    WHERE id NOT IN (
        SELECT MIN(id)
        FROM weather_data
        GROUP BY location_id, strftime('%Y-%m-%d %H:%M', timestamp)
    )
""")
deleted_dupes = cur.rowcount
print(f"🗑️  Deleted {deleted_dupes} duplicate records")

conn.commit()

# 4. Count after
cur.execute("SELECT COUNT(*) FROM weather_data")
after = cur.fetchone()[0]
print(f"✅ Records after cleanup: {after}")
print(f"✅ Total removed: {before - after}")

conn.close()
print("\n✅ Database cleanup complete!")