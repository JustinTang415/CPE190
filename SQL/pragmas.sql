-- pragmas.sql
-- Pragmas configure SQLite's performance and durability.
-- These are sane defaults for embedded devices (microSD/SSD) where you want to balance
-- safety with limited write cycles and power-loss scenarios.

PRAGMA journal_mode=WAL;         -- Write-Ahead Logging: better read concurrency, crash resistance
PRAGMA synchronous=NORMAL;       -- FULL is safest; NORMAL is faster with minor risk on power loss
PRAGMA temp_store=MEMORY;        -- temp structures in RAM to reduce disk writes
PRAGMA mmap_size=3000000000;     -- enable file-backed mmap (may help read perf on 64-bit)
PRAGMA page_size=4096;           -- align with common FS block sizes
PRAGMA foreign_keys=ON;          -- keep ON if you use FKs (none by default in this template)

-- CHANGE ME (advanced):
-- PRAGMA cache_size=-20000;       -- ~20,000 pages in KB if negative; tune based on RAM
-- PRAGMA wal_autocheckpoint=1000; -- checkpoint WAL every ~1000 pages
-- PRAGMA busy_timeout=5000;       -- wait up to 5s if the DB is locked (single-writer friendly)
