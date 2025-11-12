-- schema.sql
-- This file defines the tables and indices for telemetry.
-- You can extend these tables or add new tables for other sensors (baro, battery, RC link, etc.).
-- Keep schemas narrow (only the fields you need) and indexed by the columns you query/filter by.

-- IMU: 3-axis accelerometer as an example.
-- CHANGE ME: If you have gyroscope/magnetometer, add columns (gx, gy, gz, mx, my, mz).
CREATE TABLE violations (
  ts INTEGER NOT NULL,          -- epoch seconds (INTEGER); use ms if needed (bigger numbers)
  loc REAL NOT NULL,
  dat REAL NOT NULL
);

if()
CREATE INDEX idx_vio;

