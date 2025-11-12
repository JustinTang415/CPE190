-- schema.sql
-- This file defines the tables and indices for telemetry.
-- You can extend these tables or add new tables for other sensors (baro, battery, RC link, etc.).
-- Keep schemas narrow (only the fields you need) and indexed by the columns you query/filter by.

-- IMU: 3-axis accelerometer as an example.
-- CHANGE ME: If you have gyroscope/magnetometer, add columns (gx, gy, gz, mx, my, mz).
CREATE TABLE IF NOT EXISTS imu (
  ts INTEGER NOT NULL,          -- epoch seconds (INTEGER); use ms if needed (bigger numbers)
  ax REAL NOT NULL,
  ay REAL NOT NULL,
  az REAL NOT NULL
);

-- GPS: basic lat/lon/alt at time ts.
-- CHANGE ME: Add horiz_acc, vert_acc, fix_type, sats, speed if you need them.
CREATE TABLE IF NOT EXISTS gps (
  ts INTEGER NOT NULL,
  vert_acc REAL NOT NULL,
  alt REAL               -- nullable
);

-- Events: human-readable log entries (errors, warnings, milestones).
-- CHANGE ME: Add a component field (e.g., 'EKF', 'NAV') or code integers if you want fast filtering.
CREATE TABLE IF NOT EXISTS events (
  ts INTEGER NOT NULL,
  level TEXT NOT NULL,          -- INFO | WARN | ERROR (enforce via CHECK if you like)
  msg TEXT NOT NULL
);

-- Indices
-- Keep indices only on columns you actually filter/sort by (each index costs write overhead).
-- By default, we index time for time-ordered queries.
CREATE INDEX IF NOT EXISTS idx_imu_ts     ON imu(ts);
CREATE INDEX IF NOT EXISTS idx_gps_ts     ON gps(ts);
CREATE INDEX IF NOT EXISTS idx_events_ts  ON events(ts);

-- RETENTION (optional): Consider a periodic job to delete old rows to control DB size.
-- Example (7-day retention for imu):
-- DELETE FROM imu WHERE ts < strftime('%s','now') - 7*24*3600;
-- Add similar deletes for gps/events and VACUUM occasionally (not too often).
