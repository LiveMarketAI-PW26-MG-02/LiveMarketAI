"""
Component 9: Per-Instrument Engagement Retrieval
Retrieves the arithmetic composition of all three event-type frequency dimensions
from persisted structured relational fields for a single instrument.
"""
import sqlite3
from dataclasses import dataclass
from typing import Optional

DB_PATH = "instruments.db"

@dataclass
class InstrumentEngagementProfile:
    instrument_id: str
    visit_count: int        # integer visit frequency dimension
    watchlist_count: int    # integer watchlist frequency dimension
    order_count: int        # integer order frequency dimension
    engagement_score: int   # deterministic arithmetic composition

def retrieve_engagement_profile(conn: sqlite3.Connection,
                                instrument_id: str) -> Optional[InstrumentEngagementProfile]:
    """
    Retrieve and compose three event-type frequency dimensions
    from the activity table for the given instrument.
    """
    row = conn.execute("""
        SELECT
            COALESCE(SUM(CASE WHEN event_type=1 THEN count_increment ELSE 0 END),0),
            COALESCE(SUM(CASE WHEN event_type=2 THEN count_increment ELSE 0 END),0),
            COALESCE(SUM(CASE WHEN event_type=3 THEN count_increment ELSE 0 END),0)
        FROM activity
        WHERE instrument_id = ?
    """, (instrument_id,)).fetchone()

    if row is None:
        return None
    v, w, o = row
    return InstrumentEngagementProfile(
        instrument_id=instrument_id,
        visit_count=v,
        watchlist_count=w,
        order_count=o,
        engagement_score=v + w + o
    )

if __name__ == "__main__":
    with sqlite3.connect(DB_PATH) as conn:
        profile = retrieve_engagement_profile(conn, "INSTR_001")
        print(f"Engagement profile: {profile}")
