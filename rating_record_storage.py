"""
Component 1: Multimodal Rating Record Storage
Persists two coexisting structured numeric streams per instrument:
  - raw_rating: the decimal observation layer
  - running_total: the arithmetic accumulation layer
"""
import sqlite3
from dataclasses import dataclass
from typing import Optional

DB_PATH = "instruments.db"

@dataclass
class RatingRecord:
    instrument_id: str
    raw_rating: float          # decimal observation layer
    running_total: float       # arithmetic accumulation layer
    observation_count: int

def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ratings (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            instrument_id   TEXT    NOT NULL,
            raw_rating      REAL    NOT NULL,   -- decimal observation layer
            running_total   REAL    NOT NULL,   -- arithmetic accumulation layer
            observation_count INTEGER NOT NULL DEFAULT 1,
            created_at      TEXT    DEFAULT (datetime('now'))
        )
    """)
    conn.commit()

def insert_rating(conn: sqlite3.Connection,
                  instrument_id: str,
                  raw_rating: float) -> RatingRecord:
    """
    Persist a new decimal rating observation.
    Updates the running arithmetic accumulation layer atomically.
    """
    cursor = conn.execute(
        "SELECT COALESCE(SUM(raw_rating), 0), COUNT(*) FROM ratings WHERE instrument_id = ?",
        (instrument_id,)
    )
    prev_total, prev_count = cursor.fetchone()
    new_running_total = prev_total + raw_rating
    new_count = prev_count + 1

    conn.execute(
        """INSERT INTO ratings (instrument_id, raw_rating, running_total, observation_count)
           VALUES (?, ?, ?, ?)""",
        (instrument_id, raw_rating, new_running_total, new_count)
    )
    conn.commit()
    return RatingRecord(instrument_id, raw_rating, new_running_total, new_count)

def fetch_rating_record(conn: sqlite3.Connection,
                        instrument_id: str) -> Optional[RatingRecord]:
    """Return the latest multimodal rating record for an instrument."""
    row = conn.execute(
        """SELECT raw_rating, running_total, observation_count
           FROM ratings WHERE instrument_id = ?
           ORDER BY id DESC LIMIT 1""",
        (instrument_id,)
    ).fetchone()
    if row is None:
        return None
    return RatingRecord(instrument_id, *row)

if __name__ == "__main__":
    with sqlite3.connect(DB_PATH) as conn:
        init_db(conn)
        rec = insert_rating(conn, "INSTR_001", 4.5)
        print(f"Stored: {rec}")
        rec2 = insert_rating(conn, "INSTR_001", 3.8)
        print(f"After 2nd entry: {rec2}")
        fetched = fetch_rating_record(conn, "INSTR_001")
        print(f"Fetched: {fetched}")
