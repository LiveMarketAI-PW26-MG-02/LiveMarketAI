"""
Component 11: Chronological Multimodal Time Marker Storage
Ensures each activity entry's time-position stream value is recorded at the
precise moment of the event, preserving arithmetic lag relationships
across successive entries.
"""
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

DB_PATH = "instruments.db"

@dataclass
class TimedActivityEntry:
    instrument_id: str
    event_type: int
    count_increment: int
    event_time: str     # time-position layer (ISO UTC)
    lag_seconds: float  # arithmetic lag from previous entry (0 for first)

def record_timed_event(conn: sqlite3.Connection,
                       instrument_id: str,
                       event_type: int) -> TimedActivityEntry:
    """
    Record event with precise UTC timestamp as the time-position layer.
    Guarantees chronological ordering is inviolably preserved.
    """
    ts = datetime.utcnow().isoformat()
    conn.execute(
        """INSERT INTO activity (instrument_id, event_type, count_increment, event_time)
           VALUES (?, ?, 1, ?)""",
        (instrument_id, event_type, ts)
    )
    conn.commit()
    return TimedActivityEntry(instrument_id, event_type, 1, ts, 0.0)

def fetch_ordered_activity(conn: sqlite3.Connection,
                           instrument_id: str) -> List[TimedActivityEntry]:
    """
    Return all activity entries in chronological order with lag computation.
    """
    rows = conn.execute(
        """SELECT instrument_id, event_type, count_increment, event_time
           FROM activity WHERE instrument_id = ?
           ORDER BY event_time ASC""",
        (instrument_id,)
    ).fetchall()

    entries = []
    prev_time = None
    for r in rows:
        ts = r[3]
        if prev_time is None:
            lag = 0.0
        else:
            t1 = datetime.fromisoformat(prev_time)
            t2 = datetime.fromisoformat(ts)
            lag = (t2 - t1).total_seconds()
        entries.append(TimedActivityEntry(r[0], r[1], r[2], ts, lag))
        prev_time = ts
    return entries

if __name__ == "__main__":
    with sqlite3.connect(DB_PATH) as conn:
        ordered = fetch_ordered_activity(conn, "INSTR_001")
        for e in ordered:
            print(f"  type={e.event_type} time={e.event_time} lag={e.lag_seconds}s")
