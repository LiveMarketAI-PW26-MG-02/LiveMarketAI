"""
Component 8: Total Engagement Intensity Computation
Computes the multimodal engagement value as:
  engagement = visit_count + watchlist_count + order_count
Three coexisting integer frequency dimensions combined arithmetically.
"""
import sqlite3
from dataclasses import dataclass
from typing import List, Optional

DB_PATH = "instruments.db"
EVENT_VISIT     = 1
EVENT_WATCHLIST = 2
EVENT_ORDER     = 3

@dataclass
class EngagementIntensity:
    instrument_id: str
    visit_count: int
    watchlist_count: int
    order_count: int
    total_engagement: int   # arithmetic composition of all three layers

def compute_engagement_intensity(conn: sqlite3.Connection,
                                 instrument_id: str) -> EngagementIntensity:
    """
    Deterministically sum all three event-type frequency dimensions
    into one cross-sectional multimodal engagement value.
    """
    def get_count(event_type: int) -> int:
        row = conn.execute(
            """SELECT COALESCE(SUM(count_increment), 0)
               FROM activity
               WHERE instrument_id = ? AND event_type = ?""",
            (instrument_id, event_type)
        ).fetchone()
        return row[0]

    v = get_count(EVENT_VISIT)
    w = get_count(EVENT_WATCHLIST)
    o = get_count(EVENT_ORDER)
    return EngagementIntensity(instrument_id, v, w, o, v + w + o)

def batch_engagement_intensity(conn: sqlite3.Connection) -> List[EngagementIntensity]:
    """Compute engagement intensity for all instruments in one query."""
    rows = conn.execute("""
        SELECT
            instrument_id,
            COALESCE(SUM(CASE WHEN event_type=1 THEN count_increment ELSE 0 END),0) AS visits,
            COALESCE(SUM(CASE WHEN event_type=2 THEN count_increment ELSE 0 END),0) AS watchlists,
            COALESCE(SUM(CASE WHEN event_type=3 THEN count_increment ELSE 0 END),0) AS orders
        FROM activity
        GROUP BY instrument_id
    """).fetchall()
    return [
        EngagementIntensity(r[0], r[1], r[2], r[3], r[1]+r[2]+r[3])
        for r in rows
    ]

if __name__ == "__main__":
    with sqlite3.connect(DB_PATH) as conn:
        eng = compute_engagement_intensity(conn, "INSTR_001")
        print(f"Engagement: {eng}")
