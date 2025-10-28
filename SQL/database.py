# jetson_violation_logger.py
import os, hashlib
from datetime import datetime, timezone
import sqlite3

DB_PATH = "/home/jetson/violations.db"     # change if you prefer
IMG_DIR = "/home/jetson/violation_photos"  # make sure this exists

SCHEMA = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS violation_types (
    type_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    code        TEXT UNIQUE NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS violations (
    violation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    type_id      INTEGER NOT NULL,
    latitude     REAL NOT NULL CHECK(latitude BETWEEN -90 AND 90),
    longitude    REAL NOT NULL CHECK(longitude BETWEEN -180 AND 180),
    occurred_at  TEXT NOT NULL,
    notes        TEXT,
    FOREIGN KEY (type_id) REFERENCES violation_types(type_id) ON UPDATE CASCADE ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_violations_occurred_at ON violations(occurred_at);
CREATE INDEX IF NOT EXISTS idx_violations_type        ON violations(type_id);

CREATE TABLE IF NOT EXISTS violation_photos (
    photo_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    violation_id INTEGER NOT NULL,
    captured_at  TEXT NOT NULL,
    image_url    TEXT,
    image_blob   BLOB,
    is_primary   INTEGER NOT NULL DEFAULT 0 CHECK(is_primary IN (0,1)),
    sha256_hex   TEXT,
    FOREIGN KEY (violation_id) REFERENCES violations(violation_id) ON DELETE CASCADE,
    CONSTRAINT image_present CHECK (image_url IS NOT NULL OR image_blob IS NOT NULL)
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_violation_primary_photo
ON violation_photos(violation_id)
WHERE is_primary = 1;
"""

def connect():
    con = sqlite3.connect(DB_PATH, isolation_level=None)  # autocommit
    con.execute("PRAGMA foreign_keys=ON;")
    con.execute("PRAGMA journal_mode=WAL;")
    return con

def init_db():
    con = connect()
    with con:
        con.executescript(SCHEMA)
        con.execute("INSERT OR IGNORE INTO violation_types(code, description) VALUES(?,?)",
                    ("NO_HARD_HAT", "Missing hard hat"))
        con.execute("INSERT OR IGNORE INTO violation_types(code, description) VALUES(?,?)",
                    ("NO_VEST", "Missing safety vest"))
    con.close()

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def save_violation(type_code, lat, lon, image_path=None, notes=None):
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    con = connect()
    with con:
        type_id = con.execute("SELECT type_id FROM violation_types WHERE code=?", (type_code,)).fetchone()
        if not type_id:
            raise ValueError(f"Unknown type_code {type_code}")
        type_id = type_id[0]

        cur = con.execute(
            "INSERT INTO violations(type_id, latitude, longitude, occurred_at, notes) VALUES(?,?,?,?,?)",
            (type_id, lat, lon, ts, notes)
        )
        violation_id = cur.lastrowid

        if image_path:
            sha = sha256_file(image_path)
            con.execute(
                "INSERT INTO violation_photos(violation_id, captured_at, image_url, is_primary, sha256_hex) "
                "VALUES(?,?,?,?,?)",
                (violation_id, ts, f"file://{image_path}", 1, sha)
            )
    con.close()
    return violation_id

if __name__ == "__main__":
    os.makedirs(IMG_DIR, exist_ok=True)
    init_db()

    # OPTIONAL: capture a frame with OpenCV if available on your Jetson
    # If OpenCV isn't installed, comment out this block and provide an existing image path instead.
    try:
        import cv2
        cam = cv2.VideoCapture(0)  # might need a GStreamer pipeline on Jetson; this works for most USB cams
        ok, frame = cam.read()
        cam.release()
        if ok:
            img_path = os.path.join(IMG_DIR, f"violation_{int(datetime.now().timestamp())}.jpg")
            cv2.imwrite(img_path, frame)
        else:
            img_path = None
    except Exception:
        img_path = None  # fallback if OpenCV not available/configured

    v_id = save_violation(
        type_code="NO_HARD_HAT",
        lat=38.559, lon=-121.424,
        image_path=img_path,
        notes="Auto-detected on site A by Jetson Nano"
    )
    print(f"Saved violation {v_id}, image={img_path}")