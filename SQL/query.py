#!/usr/bin/env python3
# query.py
# Example read patterns:
# - List a few most recent IMU rows
# - Show latest GPS sample
# - Show a recent event log
#
# WHERE TO CHANGE:
# - Add WHERE clauses for time windows (e.g., the last 10 seconds).
# - Join across tables if you add new sensors.
# - Create indices in schema.sql for columns you filter by.
#
# Performance tips:
# - LIMIT keeps UI snappy on dashboards.
# - For historical analysis, page results in chunks.
import sqlite3, pathlib

DB_PATH = pathlib.Path("flight.db")

def main():
    conn = sqlite3.connect(DB_PATH)
    try:
        print("Recent IMU (5 rows):")
        for row in conn.execute(
            "SELECT ts, ax, ay, az FROM imu ORDER BY ts DESC LIMIT 5"
        ):
            print(row)

        print("\nLatest GPS:")
        row = conn.execute(
            "SELECT ts, lat, lon, alt FROM gps ORDER BY ts DESC LIMIT 1"
        ).fetchone()
        print(row)

        print("\nRecent events:")
        for row in conn.execute(
            "SELECT ts, level, msg FROM events ORDER BY ts DESC LIMIT 5"
        ):
            print(row)

        # CHANGE ME: Example time-window query (last 10 seconds):
        # for row in conn.execute(
        #   """SELECT ts, ax, ay, az FROM imu
        #        WHERE ts >= strftime('%s','now') - 10
        #        ORDER BY ts DESC"""
        # ):
        #     print(row)

    finally:
        conn.close()

if __name__ == "__main__":
    main()
