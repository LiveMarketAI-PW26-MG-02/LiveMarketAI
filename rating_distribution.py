"""
Component 3: Rating Distribution Computation
Produces the multimodal dispersion layer:
  frequency counts per arithmetically defined scoring band.
Bands: [0-1), [1-2), [2-3), [3-4), [4-5]
"""
import sqlite3
from dataclasses import dataclass
from typing import List

DB_PATH = "instruments.db"

BANDS = [
    (0.0, 1.0, "0-1"),
    (1.0, 2.0, "1-2"),
    (2.0, 3.0, "2-3"),
    (3.0, 4.0, "3-4"),
    (4.0, 5.0, "4-5"),
]

@dataclass
class BandFrequency:
    band_label: str
    lower: float
    upper: float
    frequency_count: int

@dataclass
class RatingDistribution:
    instrument_id: str
    bands: List[BandFrequency]
    total_observations: int

def compute_distribution(conn: sqlite3.Connection,
                         instrument_id: str) -> RatingDistribution:
    """
    Derive frequency count per scoring band from persisted decimal rating values.
    """
    bands = []
    total = 0
    for lower, upper, label in BANDS:
        if upper == 5.0:   # inclusive upper bound for max band
            row = conn.execute(
                """SELECT COUNT(*) FROM ratings
                   WHERE instrument_id = ?
                   AND raw_rating >= ? AND raw_rating <= ?""",
                (instrument_id, lower, upper)
            ).fetchone()
        else:
            row = conn.execute(
                """SELECT COUNT(*) FROM ratings
                   WHERE instrument_id = ?
                   AND raw_rating >= ? AND raw_rating < ?""",
                (instrument_id, lower, upper)
            ).fetchone()
        count = row[0] if row else 0
        total += count
        bands.append(BandFrequency(label, lower, upper, count))

    return RatingDistribution(instrument_id, bands, total)

if __name__ == "__main__":
    with sqlite3.connect(DB_PATH) as conn:
        dist = compute_distribution(conn, "INSTR_001")
        print(f"Distribution for {dist.instrument_id} (n={dist.total_observations}):")
        for b in dist.bands:
            print(f"  [{b.band_label}]: {b.frequency_count}")
