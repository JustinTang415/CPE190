#!/usr/bin/env python3
# Initialize the SQLite database with safe defaults for embedded devices.
# What this script does:
# 1) Connects (or creates) flight.db in the current directory.
# 2) Applies pragmas from pragmas.sql to tune performance/durability.
# 3) Applies schema from schema.sql to create your tables/indices.
#
# Where to customize:
# - DB location: change DB_PATH
# - Pragmas: edit pragmas.sql
# - Tables/indices: edit schema.sql
#
# If you rename the DB or want to place it on a different storage device
# (e.g., SSD vs microSD), change DB_PATH accordingly.
import sqlite3, pathlib, sys

# CHANGE ME: If you want to place the DB elsewhere (e.g., /data/flight.db or /mnt/ssd/flight.db)
DB_PATH = pathlib.Path("flight.db")

def run_script(conn, path):
    """Execute a .sql script file (pragmas or schema)."""
    with open(path, "r", encoding="utf-8") as f:
        conn.executescript(f.read())

def main():
    # Connect (creates file if missing).
    conn = sqlite3.connect(DB_PATH)
    try:
        # Apply pragmas first (sets WAL, synchronous, etc.).
        run_script(conn, "pragmas.sql")
        # Then ensure tables/indices exist.
        run_script(conn, "schema.sql")
        conn.commit()

        # Inspect and print some active settings so you can verify them.
        print(f"Initialized database at {DB_PATH.resolve()}")
        jm = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        syn = conn.execute("PRAGMA synchronous;").fetchone()[0]
        print("journal_mode:", jm)
        print("synchronous:", syn)

        # CHANGE ME: Generate additional tables programmatically here if needed.
    finally:
        conn.close()

if __name__ == "__main__":
    sys.exit(main())
