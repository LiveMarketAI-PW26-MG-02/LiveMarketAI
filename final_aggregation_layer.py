"""
Component 15: Final Multimodal Interaction Aggregation Layer
Converges five coexisting structured numeric dimensional layers per instrument:
  1. decimal_rating_total  - decimal rating observation stream (SUM)
  2. visit_count           - integer visit frequency dimension
  3. watchlist_count       - integer watchlist frequency dimension
  4. order_count           - integer order frequency dimension
  5. instrument_id         - instrument identifier stream
All values are arithmetic results of deterministic operations on persisted relational fields.
"""
import sqlite3
from dataclasses import dataclass
from typing import List, Optional

DB_PATH = "instruments.db"

@dataclass
class MultimodalAggregateRecord:
    instrument_id: str
    decimal_rating_total: float  # SUM of raw decimal observations
    mean_rating: Optional[float] # arithmetic mean layer
    rating_count: int
    visit_count: int
    watchlist_count: int
    order_count: int
    total_engagement: int        # v + w + o
    composite_score: float       # mean_rating * log1p(engagement) -- combined signal

def build_aggregate_record(conn: sqlite3.Connection,
                           instrument_id: str) -> MultimodalAggregateRecord:
    """
    Deterministically converge all five dimensional layers from rating + activity tables.
    """
    import math
    r_row = conn.execute(
        """SELECT COALESCE(SUM(raw_rating),0), AVG(raw_rating), COUNT(*)
           FROM ratings WHERE instrument_id = ?""",
        (instrument_id,)
    ).fetchone()
    rating_total, mean_rating, rating_count = r_row

    a_row = conn.execute("""
        SELECT
            COALESCE(SUM(CASE WHEN event_type=1 THEN count_increment ELSE 0 END),0),
            COALESCE(SUM(CASE WHEN event_type=2 THEN count_increment ELSE 0 END),0),
            COALESCE(SUM(CASE WHEN event_type=3 THEN count_increment ELSE 0 END),0)
        FROM activity WHERE instrument_id = ?
    """, (instrument_id,)).fetchone()
    v, w, o = a_row if a_row else (0, 0, 0)
    engagement = v + w + o
    mean = round(mean_rating, 6) if mean_rating else 0.0
    composite = round(mean * math.log1p(engagement), 6) if engagement > 0 else 0.0

    return MultimodalAggregateRecord(
        instrument_id=instrument_id,
        decimal_rating_total=round(rating_total, 6),
        mean_rating=mean,
        rating_count=rating_count,
        visit_count=v,
        watchlist_count=w,
        order_count=o,
        total_engagement=engagement,
        composite_score=composite
    )

def build_all_aggregate_records(conn: sqlite3.Connection) -> List[MultimodalAggregateRecord]:
    """Build the final aggregation layer for every instrument."""
    ids = conn.execute(
        "SELECT DISTINCT instrument_id FROM (SELECT instrument_id FROM ratings UNION SELECT instrument_id FROM activity)"
    ).fetchall()
    return [build_aggregate_record(conn, row[0]) for row in ids]

if __name__ == "__main__":
    with sqlite3.connect(DB_PATH) as conn:
        records = build_all_aggregate_records(conn)
        for rec in records:
            print(rec)
