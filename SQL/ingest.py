#!/usr/bin/env python3
# ingest.py
# Example write patterns:
# - Batch many IMU rows inside a single transaction (fast, durable enough with synchronous=NORMAL).
# - Write single GPS sample.
# - Write a human-readable event.
#
# WHERE TO CHANGE:
# - Adjust batch sizes / sensor schemas
# - Wrap high-rate sensors in transactions (BEGIN ... COMMIT) every N rows or milliseconds.
# - Consider a producer/consumer queue if multiple processes feed data.
#
# Durability & wear tips:
# - Keep WAL mode ON (set in pragmas.sql).
# - Use transactions to group writes and reduce fsyncs.
# - On microSD, prefer fewer, larger transactions.
import sqlite3, pathlib, time, random, contextlib

DB_PATH = pathlib.Path("flight.db")

def insert_imu_batch(conn, rows):
    """Insert many IMU rows quickly. rows = [(ts, ax, ay, az), ...]"""
    conn.executemany(
        "INSERT INTO imu (ts, ax, ay, az) VALUES (?, ?, ?, ?)",
        rows
    )

def insert_gps(conn, ts, lat, lon, alt=None):
    """Insert one GPS point (keep it simple for low-rate sensors)."""
    conn.execute(
        "INSERT INTO gps (ts, lat, lon, alt) VALUES (?, ?, ?, ?)",
        (ts, lat, lon, alt)
    )

def log_event(conn, level, msg):
    """Write a human-readable log entry. Levels: INFO/WARN/ERROR."""
    conn.execute(
        "INSERT INTO events (ts, level, msg) VALUES (strftime('%s','now'), ?, ?)",
        (level, msg)
    )

def main():
    # isolation_level=None lets us control transactions manually with BEGIN/COMMIT/ROLLBACK
    conn = sqlite3.connect(DB_PATH, isolation_level=None)
    try:
        # 1) Batch an IMU burst in a single transaction.
        # CHANGE ME: Tune batch size by sample rate and acceptable latency.
        conn.execute("BEGIN")
        now = int(time.time())
        batch = []
        for i in range(100):  # e.g., 100 samples per burst
            ts = now + i
            ax, ay, az = (
                random.uniform(-1, 1),
                random.uniform(-1, 1),
                9.8 + random.uniform(-0.05, 0.05)
            )
            batch.append((ts, ax, ay, az))
        insert_imu_batch(conn, batch)
        conn.execute("COMMIT")

        # 2) Single GPS sample (low rate).
        insert_gps(conn, int(time.time()), 38.58, -121.49, 12.3)

        # 3) Event (log line).
        log_event(conn, "INFO", "Inserted sample telemetry batch")

        conn.commit()  # safe to call (no-op if nothing pending)

        print("Inserted sample data.")
    except Exception as e:
        # Roll back any in-flight transaction on error to avoid partial writes.
        with contextlib.suppress(Exception):
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    main()
