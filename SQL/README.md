# Jetson Nano SQLite Template (Heavily Commented)

This template is designed for Jetson Nano (Ubuntu aarch64) and any small Linux device.
It shows a safe, simple pattern for logging telemetry with SQLite, using WAL mode, batched writes,
and clear places to expand your database as your drone grows (more sensors, higher rates, etc.).

If you only skim one file, start with `init_db.py` — it explains the core choices and how to evolve them.

## Project layout
```
jetson-sqlite-template-commented/
├── README.md
├── schema.sql
├── pragmas.sql
├── init_db.py
├── ingest.py
├── query.py
└── requirements.txt
```

## Quick start (on the Nano)
```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip sqlite3

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 init_db.py    # Creates/initializes flight.db with pragmas + schema
python3 ingest.py     # Writes a batch of example rows
python3 query.py      # Reads them back
```

## Where to change things
- Tables/columns → schema.sql
- Performance/durability settings → pragmas.sql
- DB path & startup → init_db.py (DB_PATH)
- Write patterns (batch/transactions) → ingest.py
- Read patterns (queries/indices) → query.py + add indices in schema.sql

Search for "CHANGE ME" comments in the code.
