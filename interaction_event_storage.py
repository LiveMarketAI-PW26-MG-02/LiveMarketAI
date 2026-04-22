"""
Component 5: Multimodal Interaction Event Storage
Stores three coexisting structured numeric streams per event:
  - event_type:   categorical integer layer (1=visit, 2=watchlist, 3=order)
  - count_increment: activity frequency stream (always 1 per event)
  - event_time:   time-position layer (ISO timestamp)
"""
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import List

DB_PATH = "instruments.db"

EVENT_VISIT     = 1
EVENT_WATCHLIST = 2
EVENT_ORDER     = 3

EVENT_LABELS = {
    EVENT_VISIT:     "visit",
    EVENT_WATCHLIST: "watchlist",
    EVENT_ORDER:     "order",
}

@dataclass
class InteractionEvent:
    instrument_id: str
    event_type: int             # categorical integer layer
    count_increment: int        # activity frequency stream dimension (=1)
    event_time: str             # time-position layer

def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS activity (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            instrument_id   TEXT    NOT NULL,
            event_type      INTEGER NOT NULL,   -- categorical integer layer
            count_increment INTEGER NOT NULL DEFAULT 1, -- frequency stream
            event_time      TEXT    NOT NULL    -- time-position layer
        )
    """)
    conn.commit()

def record_event(conn: sqlite3.Connection,
                 instrument_id: str,
                 event_type: int,
                 event_time: Optional[str] = None) -> InteractionEvent:
    """Persist one multimodal interaction event with all three numeric streams."""
    from typing import Optional
    ts = event_time or datetime.utcnow().isoformat()
    conn.execute(
        """INSERT INTO activity (instrument_id, event_type, count_increment, event_time)
           VALUES (?, ?, 1, ?)""",
        (instrument_id, event_type, ts)
    )
    conn.commit()
    return InteractionEvent(instrument_id, event_type, 1, ts)

def fetch_events(conn: sqlite3.Connection,
                 instrument_id: str) -> List[InteractionEvent]:
    rows = conn.execute(
        """SELECT instrument_id, event_type, count_increment, event_time
           FROM activity WHERE instrument_id = ? ORDER BY event_time""",
        (instrument_id,)
    ).fetchall()
    return [InteractionEvent(*r) for r in rows]

if __name__ == "__main__":
    from typing import Optional
    with sqlite3.connect(DB_PATH) as conn:
        init_db(conn)
        e1 = record_event(conn, "INSTR_001", EVENT_VISIT)
        e2 = record_event(conn, "INSTR_001", EVENT_WATCHLIST)
        e3 = record_event(conn, "INSTR_001", EVENT_ORDER)
        print(f"Recorded: {e1}, {e2}, {e3}")
        events = fetch_events(conn, "INSTR_001")
        for e in events:
            print(f"  [{EVENT_LABELS[e.event_type]}] @ {e.event_time}")
