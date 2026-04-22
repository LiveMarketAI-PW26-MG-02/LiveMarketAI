"""
Component 13: Unified Instrument Engagement Endpoint
Combines two coexisting structured numeric layers:
  - mean_rating:       arithmetic mean of the decimal rating observation stream
  - total_engagement:  integrated level of the activity frequency stream
into one unified authoritative multimodal dimensional record per instrument.
"""
import sqlite3
from dataclasses import dataclass
from typing import Optional

DB_PATH = "instruments.db"

@dataclass
class UnifiedInstrumentProfile:
    instrument_id: str
    mean_rating: Optional[float]   # arithmetic mean dimensional layer
    rating_count: int
    visit_count: int
    watchlist_count: int
    order_count: int
    total_engagement: int          # integrated activity frequency level

def get_unified_profile(conn: sqlite3.Connection,
                        instrument_id: str) -> UnifiedInstrumentProfile:
    """
    Join rating and activity tables to produce the unified multimodal profile.
    """
    # Rating layer
    rating_row = conn.execute(
        """SELECT AVG(raw_rating), COUNT(*) FROM ratings WHERE instrument_id = ?""",
        (instrument_id,)
    ).fetchone()
    mean_rating = round(rating_row[0], 6) if rating_row[0] else None
    rating_count = rating_row[1] or 0

    # Activity layer
    activity_row = conn.execute("""
        SELECT
            COALESCE(SUM(CASE WHEN event_type=1 THEN count_increment ELSE 0 END),0),
            COALESCE(SUM(CASE WHEN event_type=2 THEN count_increment ELSE 0 END),0),
            COALESCE(SUM(CASE WHEN event_type=3 THEN count_increment ELSE 0 END),0)
        FROM activity WHERE instrument_id = ?
    """, (instrument_id,)).fetchone()
    v, w, o = activity_row if activity_row else (0, 0, 0)

    return UnifiedInstrumentProfile(
        instrument_id=instrument_id,
        mean_rating=mean_rating,
        rating_count=rating_count,
        visit_count=v,
        watchlist_count=w,
        order_count=o,
        total_engagement=v + w + o
    )

if __name__ == "__main__":
    with sqlite3.connect(DB_PATH) as conn:
        profile = get_unified_profile(conn, "INSTR_001")
        print(f"Unified profile: {profile}")
