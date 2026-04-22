"""
Component 4: Rating Participation Frequency Computation
Computes the activity frequency stream dimension:
  integer count of recorded decimal evaluation entries per instrument.
Used to identify sequential lag depth in the rating record.
"""
import sqlite3
from dataclasses import dataclass
from typing import List, Optional

DB_PATH = "instruments.db"

@dataclass
class ParticipationFrequency:
    instrument_id: str
    rating_count: int           # integer count stream
    lag_depth: int              # sequential lag depth (reverse index of latest entry)

def get_participation_frequency(conn: sqlite3.Connection,
                                instrument_id: str) -> Optional[ParticipationFrequency]:
    """Return the integer count stream value for one instrument."""
    row = conn.execute(
        "SELECT COUNT(*) FROM ratings WHERE instrument_id = ?",
        (instrument_id,)
    ).fetchone()
    if not row or row[0] == 0:
        return None
    count = row[0]
    return ParticipationFrequency(
        instrument_id=instrument_id,
        rating_count=count,
        lag_depth=count - 1   # zero-indexed lag from latest
    )

def all_participation_frequencies(conn: sqlite3.Connection) -> List[ParticipationFrequency]:
    """Return participation frequencies for all instruments, descending."""
    rows = conn.execute(
        """SELECT instrument_id, COUNT(*) as cnt
           FROM ratings
           GROUP BY instrument_id
           ORDER BY cnt DESC"""
    ).fetchall()
    return [
        ParticipationFrequency(r[0], r[1], r[1] - 1)
        for r in rows
    ]

if __name__ == "__main__":
    with sqlite3.connect(DB_PATH) as conn:
        pf = get_participation_frequency(conn, "INSTR_001")
        print(f"Participation: {pf}")
        all_pf = all_participation_frequencies(conn)
        print(f"All: {all_pf}")
