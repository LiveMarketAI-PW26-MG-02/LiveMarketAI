"""
Component 12: Period-Scoped Engagement Log Retrieval
Returns a temporally bounded multimodal window of the activity record:
  visit_count, watchlist_count, order_count within [start_time, end_time].
"""
import sqlite3
from dataclasses import dataclass
from typing import Optional

DB_PATH = "instruments.db"

@dataclass
class PeriodEngagementWindow:
    instrument_id: str
    start_time: str
    end_time: str
    visit_count: int
    watchlist_count: int
    order_count: int
    window_total: int   # arithmetic composition within the period

def retrieve_period_engagement(conn: sqlite3.Connection,
                               instrument_id: str,
                               start_time: str,
                               end_time: str) -> PeriodEngagementWindow:
    """
    Retrieve the three frequency dimensions for the bounded time window.
    Arithmetic relationships across dimensions are preserved internally.
    """
    row = conn.execute("""
        SELECT
            COALESCE(SUM(CASE WHEN event_type=1 THEN count_increment ELSE 0 END),0),
            COALESCE(SUM(CASE WHEN event_type=2 THEN count_increment ELSE 0 END),0),
            COALESCE(SUM(CASE WHEN event_type=3 THEN count_increment ELSE 0 END),0)
        FROM activity
        WHERE instrument_id = ?
          AND event_time >= ?
          AND event_time <= ?
    """, (instrument_id, start_time, end_time)).fetchone()

    v, w, o = row if row else (0, 0, 0)
    return PeriodEngagementWindow(
        instrument_id=instrument_id,
        start_time=start_time,
        end_time=end_time,
        visit_count=v,
        watchlist_count=w,
        order_count=o,
        window_total=v + w + o
    )

if __name__ == "__main__":
    start = "2024-01-01T00:00:00"
    end   = "2099-12-31T23:59:59"
    with sqlite3.connect(DB_PATH) as conn:
        window = retrieve_period_engagement(conn, "INSTR_001", start, end)
        print(f"Period window: {window}")
