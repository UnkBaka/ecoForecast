# seed_locations.py
from models.database import get_connection


def seed():
    cities = [
        ("Kuala Lumpur", 3.1478, 101.6953),
        ("George Town", 5.4144, 100.3292),
        ("Johor Bahru", 1.4556, 103.7611),
        ("Ipoh", 4.5972, 101.0750),
        ("Kota Kinabalu", 5.9750, 116.0725),
        ("Kuching", 1.5575, 110.3439),
        ("Malacca City", 2.1944, 102.2486),
        ("Shah Alam", 3.0722, 101.5167),
        ("Petaling Jaya", 3.0972, 101.6444),
        ("Seremban", 2.7222, 101.9417)
    ]

    conn = get_connection()
    cur = conn.cursor()
    cur.executemany("INSERT OR IGNORE INTO locations (name, lat, lon) VALUES (?, ?, ?)", cities)
    conn.commit()
    conn.close()
    print("✅ 10 Malaysian cities seeded with coordinates!")


if __name__ == "__main__":
    seed()