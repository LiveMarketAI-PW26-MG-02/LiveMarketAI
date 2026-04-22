"""
Component 10: Instrument-Level Interaction Grouping
Deterministically accumulates the three integer event-type frequency dimensions
across all activity entries sharing the same instrument_id into one unified grouped record.
"""
import sqlite3
from dataclasses import dataclass
from typing import List

DB_PATH = "instruments.db"

@dataclass
class GroupedInteractionRecord:
    instrument_id: str
    visit_count: int
    watchlist_count: int
    order_count: int
    total_events: int

def group_interactions_by_instrument(conn: sqlite3.Connection) -> List[GroupedInteractionRecord]:
    """
    GROUP BY instrument_id to unify all three numeric event-type
    frequency dimensions into one per-instrument relational record.
    """
    rows = conn.execute("""
        SELECT
            instrument_id,
            COALESCE(SUM(CASE WHEN event_type=1 THEN count_increment ELSE 0 END),0) AS visits,
            COALESCE(SUM(CASE WHEN event_type=2 THEN count_increment ELSE 0 END),0) AS watchlists,
            COALESCE(SUM(CASE WHEN event_type=3 THEN count_increment ELSE 0 END),0) AS orders
        FROM activity
        GROUP BY instrument_id
        ORDER BY instrument_id
    """).fetchall()
    return [
        GroupedInteractionRecord(r[0], r[1], r[2], r[3], r[1]+r[2]+r[3])
        for r in rows
    ]

def get_grouped_record(conn: sqlite3.Connection,
                       instrument_id: str) -> GroupedInteractionRecord:
    """Return the unified grouped multimodal observation for one instrument."""
    row = conn.execute("""
        SELECT
            COALESCE(SUM(CASE WHEN event_type=1 THEN count_increment ELSE 0 END),0),
            COALESCE(SUM(CASE WHEN event_type=2 THEN count_increment ELSE 0 END),0),
            COALESCE(SUM(CASE WHEN event_type=3 THEN count_increment ELSE 0 END),0)
        FROM activity WHERE instrument_id = ?
    """, (instrument_id,)).fetchone()
    v, w, o = row if row else (0, 0, 0)
    return GroupedInteractionRecord(instrument_id, v, w, o, v+w+o)

if __name__ == "__main__":
    with sqlite3.connect(DB_PATH) as conn:
        all_groups = group_interactions_by_instrument(conn)
        for g in all_groups:
            print(g)
