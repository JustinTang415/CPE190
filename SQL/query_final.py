#!/usr/bin/env python3
# ============================================================================
# query_final.py — FINAL VERSION (Fully Commented)
#
# PURPOSE:
#   • This script manages the violations database used by your drone.
#   • The drone ONLY logs violations (NO VEST / NO HARD HAT).
#   • If the drone does not detect a violation, it does NOT log anything.
#   • Missing GPS → location stored as "0,0"
#   • Missing timestamp → time_data stored as "0"
#   • Status is ALWAYS "VIOLATION" and placed at the far right.
#
#   Two commands:
#       python3 query.py add   -> adds a violation
#       python3 query.py list  -> lists all violations
# ============================================================================


import argparse        # Handles the command-line interface (add/list commands)
import sqlite3         # SQLite library used to store and retrieve violation records
import pathlib         # Used to handle the database file path in a safe way


# ----------------------------------------------------------------------------
# DATABASE LOCATION
# ----------------------------------------------------------------------------

# DB_PATH:
#   This is where the SQLite file is stored on the Jetson Nano.
#   If the file does not exist, SQLite will automatically create it.
DB_PATH = pathlib.Path("/home/jetson/violations.db")


# ----------------------------------------------------------------------------
# SQL SCHEMA — Defines the "violations" table
# ----------------------------------------------------------------------------

# SCHEMA_SQL:
#   • Creates the table ONLY if it does not exist (avoids errors)
#   • Columns:
#       V_ID      integer primary key — unique record number
#       type      text — type of violation ("NO_VEST", "NO_HARD_HAT")
#       location  text — stored as "lat,lon"
#       time_data text — timestamp ("0" if missing)
#       status    text — always "VIOLATION"
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS violations (
    V_ID      INTEGER PRIMARY KEY AUTOINCREMENT,
    type      TEXT NOT NULL,
    location  TEXT NOT NULL DEFAULT '0,0',
    time_data TEXT NOT NULL DEFAULT '0',
    status    TEXT NOT NULL DEFAULT 'VIOLATION'
);
"""


# ----------------------------------------------------------------------------
# CONNECT TO DATABASE
# ----------------------------------------------------------------------------
def connect(db_path: pathlib.Path):
    """
    Connects to the SQLite database.
    If the file doesn't exist, SQLite creates it automatically.

    row_factory = sqlite3.Row
        Makes rows behave like dictionaries,
        so you can do row["type"] instead of row[1].
    """
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


# ----------------------------------------------------------------------------
# ENSURE TABLE EXISTS
# ----------------------------------------------------------------------------
def ensure_schema(con):
    """
    Executes the CREATE TABLE statement if the table does not exist.
    Safe to run every time — it will never erase your data.
    """
    con.executescript(SCHEMA_SQL)
    con.commit()


# ----------------------------------------------------------------------------
# PRETTY-PRINT TABLE OUTPUT
# ----------------------------------------------------------------------------
def print_rows(rows):
    """
    Prints database rows in a clean column-aligned table format.
    If no rows exist → prints "(no rows)".
    """
    data = [dict(r) for r in rows]

    # No records in table
    if not data:
        print("(no rows)")
        return

    # Auto-resize columns to fit longest value
    widths = {k: max(len(k), max(len(str(d[k])) for d in data)) for k in data[0].keys()}

    # Build header row
    header = " | ".join(k.ljust(widths[k]) for k in widths)
    print(header)
    print("-" * len(header))

    # Print each row
    for d in data:
        print(" | ".join(str(d[k]).ljust(widths[k]) for k in widths))


# ----------------------------------------------------------------------------
# LIST ALL VIOLATIONS
# ----------------------------------------------------------------------------
def list_all(con):
    """
    Retrieves all violations from newest to oldest.
    SELECT returns:
        V_ID, type, location, time_data, status
    """
    return con.execute(
        "SELECT V_ID, type, location, time_data, status FROM violations ORDER BY V_ID DESC"
    ).fetchall()


# ----------------------------------------------------------------------------
# ADD A VIOLATION
# ----------------------------------------------------------------------------
def add_violation(con, vtype, lat, lon, time_data):
    """
    Adds a violation detected by the drone.

    PARAMETERS:
        vtype      — violation type (NO_VEST, NO_HARD_HAT)
        lat        — latitude (float or None)
        lon        — longitude (float or None)
        time_data  — timestamp string or None

    RULES:
        • If GPS missing → lat=0, lon=0
        • If time missing → time_data="0"
        • Drone logs ONLY real violations → ALWAYS status="VIOLATION"
    """

    # --- LATITUDE ---
    try:
        lat = float(lat) if lat is not None else 0
    except:
        lat = 0

    # --- LONGITUDE ---
    try:
        lon = float(lon) if lon is not None else 0
    except:
        lon = 0

    # Build "lat,lon" tuple string
    location = f"{lat},{lon}"

    # --- TIME (timestamp) ---
    if not time_data or str(time_data).strip() == "":
        time_data = "0"

    # --- STATUS ---
    status = "VIOLATION"   # Only real violations are logged

    # INSERT ROW
    con.execute(
        "INSERT INTO violations(type, location, time_data, status) VALUES (?, ?, ?, ?)",
        (vtype, location, time_data, status)
    )
    con.commit()

    # Return the inserted row
    return con.execute(
        "SELECT V_ID, type, location, time_data, status FROM violations ORDER BY V_ID DESC LIMIT 1"
    ).fetchall()


# ----------------------------------------------------------------------------
# MAIN — COMMAND LINE INTERFACE
# ----------------------------------------------------------------------------
def main():
    # Create parser
    ap = argparse.ArgumentParser(description="Drone Violation Logger")

    # Create sub-commands: add, list
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ---- ADD COMMAND ----
    p_add = sub.add_parser("add", help="Add a violation detected by the drone")
    p_add.add_argument("type", type=str, help="Type of violation (NO_VEST, NO_HARD_HAT)")
    p_add.add_argument("--lat", type=float, default=None, help="Latitude")
    p_add.add_argument("--lon", type=float, default=None, help="Longitude")
    p_add.add_argument("--time", type=str, default=None, help="Timestamp string")
    p_add.add_argument("--db", type=pathlib.Path, default=DB_PATH)

    # ---- LIST COMMAND ----
    p_list = sub.add_parser("list", help="List all violations")
    p_list.add_argument("--db", type=pathlib.Path, default=DB_PATH)

    # Parse user input
    args = ap.parse_args()

    # Connect to database
    con = connect(args.db)
    ensure_schema(con)

    try:

        # If user typed "add"
        if args.cmd == "add":
            rows = add_violation(con, args.type, args.lat, args.lon, args.time)
            print_rows(rows)
            return

        # If user typed "list"
        if args.cmd == "list":
            print_rows(list_all(con))
            return

    finally:
        # Always close DB
        con.close()


# ----------------------------------------------------------------------------
# SCRIPT ENTRY POINT
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
