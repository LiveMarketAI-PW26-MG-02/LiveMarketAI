"""
Component 7: Watchlist Addition Frequency Computation
Computes the running arithmetic summation of the integer count-increment stream
of watchlist inclusion events (event_type=2) per instrument.
"""
import sqlite3
from dataclasses import dataclass
from typing import List, Optional

DB_PATH = "instruments.db"
EVENT_WATCHLIST = 2

@dataclass
class WatchlistFrequency:
    instrument_id: str
    watchlist_count: int   # integrated level of passive investor interest

def get_watchlist_frequency(conn: sqlite3.Connection,
                             instrument_id: str) -> Optional[WatchlistFrequency]:
    """Return the deterministic watchlist total for one instrument."""
    row = conn.execute(
        """SELECT SUM(count_increment)
           FROM activity
           WHERE instrument_id = ? AND event_type = ?""",
        (instrument_id, EVENT_WATCHLIST)
    ).fetchone()
    count = row[0] or 0
    return WatchlistFrequency(instrument_id, count)

def all_watchlist_frequencies(conn: sqlite3.Connection) -> List[WatchlistFrequency]:
    """Batch: watchlist accumulation for all instruments."""
    rows = conn.execute(
        """SELECT instrument_id, SUM(count_increment) AS watchlist_count
           FROM activity
           WHERE event_type = ?
           GROUP BY instrument_id
           ORDER BY watchlist_count DESC""",
        (EVENT_WATCHLIST,)
    ).fetchall()
    return [WatchlistFrequency(r[0], r[1] or 0) for r in rows]

if __name__ == "__main__":
    with sqlite3.connect(DB_PATH) as conn:
        wf = get_watchlist_frequency(conn, "INSTR_001")
        print(f"Watchlist frequency: {wf}")
        all_wf = all_watchlist_frequencies(conn)
        print(f"All instruments: {all_wf}")
