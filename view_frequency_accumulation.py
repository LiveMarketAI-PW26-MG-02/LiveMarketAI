"""
Component 6: View Frequency Accumulation
Accumulates the integer count-increment stream of visit events (event_type=1)
into a running visit frequency total per instrument.
"""
import sqlite3
from dataclasses import dataclass
from typing import List, Optional

DB_PATH = "instruments.db"
EVENT_VISIT = 1

@dataclass
class ViewFrequency:
    instrument_id: str
    visit_count: int    # running arithmetic total of count_increment stream

def get_view_frequency(conn: sqlite3.Connection,
                       instrument_id: str) -> Optional[ViewFrequency]:
    """Return the deterministic running visit total for one instrument."""
    row = conn.execute(
        """SELECT SUM(count_increment)
           FROM activity
           WHERE instrument_id = ? AND event_type = ?""",
        (instrument_id, EVENT_VISIT)
    ).fetchone()
    count = row[0] or 0
    return ViewFrequency(instrument_id, count)

def all_view_frequencies(conn: sqlite3.Connection) -> List[ViewFrequency]:
    """Return visit frequency totals for all instruments."""
    rows = conn.execute(
        """SELECT instrument_id, SUM(count_increment) as visit_count
           FROM activity
           WHERE event_type = ?
           GROUP BY instrument_id
           ORDER BY visit_count DESC""",
        (EVENT_VISIT,)
    ).fetchall()
    return [ViewFrequency(r[0], r[1] or 0) for r in rows]

if __name__ == "__main__":
    with sqlite3.connect(DB_PATH) as conn:
        vf = get_view_frequency(conn, "INSTR_001")
        print(f"View frequency: {vf}")
        all_vf = all_view_frequencies(conn)
        print(f"All instruments: {all_vf}")
