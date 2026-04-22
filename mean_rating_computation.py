"""
Component 2: Mean Rating Computation
Computes the arithmetic mean dimensional layer:
  mean = running_total / observation_count
"""
import sqlite3
from dataclasses import dataclass
from typing import Optional

DB_PATH = "instruments.db"

@dataclass
class MeanRatingResult:
    instrument_id: str
    running_total: float
    observation_count: int
    mean_rating: float          # arithmetic quotient layer

def compute_mean_rating(conn: sqlite3.Connection,
                        instrument_id: str) -> Optional[MeanRatingResult]:
    """
    Derive the arithmetic mean from the accumulated decimal rating stream.
    mean = SUM(raw_rating) / COUNT(observations)
    """
    row = conn.execute(
        """SELECT SUM(raw_rating), COUNT(*)
           FROM ratings
           WHERE instrument_id = ?""",
        (instrument_id,)
    ).fetchone()
    if row is None or row[1] == 0:
        return None
    total, count = row
    return MeanRatingResult(
        instrument_id=instrument_id,
        running_total=total,
        observation_count=count,
        mean_rating=round(total / count, 6)
    )

def batch_mean_ratings(conn: sqlite3.Connection) -> list:
    """Return mean rating for every instrument in the ratings table."""
    rows = conn.execute(
        """SELECT instrument_id,
                  SUM(raw_rating)       AS running_total,
                  COUNT(*)              AS observation_count,
                  AVG(raw_rating)       AS mean_rating
           FROM ratings
           GROUP BY instrument_id"""
    ).fetchall()
    return [
        MeanRatingResult(r[0], r[1], r[2], round(r[3], 6))
        for r in rows
    ]

if __name__ == "__main__":
    with sqlite3.connect(DB_PATH) as conn:
        result = compute_mean_rating(conn, "INSTR_001")
        print(f"Mean rating: {result}")
        all_means = batch_mean_ratings(conn)
        print(f"All instruments: {all_means}")
