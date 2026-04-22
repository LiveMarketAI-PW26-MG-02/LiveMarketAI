"""
Component 14: Engagement Intensity Metric Computation
Computes the multimodal activity level at the current recorded observation point:
  activity_level = visit_count + watchlist_count + order_count
as a deterministic arithmetic composition of all three coexisting event-type dimensions.
"""
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

DB_PATH = "instruments.db"

@dataclass
class EngagementIntensityMetric:
    instrument_id: str
    snapshot_time: str          # current recorded observation point
    visit_count: int
    watchlist_count: int
    order_count: int
    activity_level: int         # cross-sectional multimodal engagement value

def compute_intensity_at_point(conn: sqlite3.Connection,
                               instrument_id: str,
                               as_of: Optional[str] = None) -> EngagementIntensityMetric:
    """
    Compute activity level up to a given time-point (defaults to now).
    """
    snapshot = as_of or datetime.utcnow().isoformat()
    row = conn.execute("""
        SELECT
            COALESCE(SUM(CASE WHEN event_type=1 THEN count_increment ELSE 0 END),0),
            COALESCE(SUM(CASE WHEN event_type=2 THEN count_increment ELSE 0 END),0),
            COALESCE(SUM(CASE WHEN event_type=3 THEN count_increment ELSE 0 END),0)
        FROM activity
        WHERE instrument_id = ? AND event_time <= ?
    """, (instrument_id, snapshot)).fetchone()
    v, w, o = row if row else (0, 0, 0)
    return EngagementIntensityMetric(
        instrument_id=instrument_id,
        snapshot_time=snapshot,
        visit_count=v,
        watchlist_count=w,
        order_count=o,
        activity_level=v + w + o
    )

if __name__ == "__main__":
    with sqlite3.connect(DB_PATH) as conn:
        metric = compute_intensity_at_point(conn, "INSTR_001")
        print(f"Intensity metric: {metric}")
        # Historical snapshot example
        historical = compute_intensity_at_point(conn, "INSTR_001", "2024-06-01T00:00:00")
        print(f"Historical: {historical}")
