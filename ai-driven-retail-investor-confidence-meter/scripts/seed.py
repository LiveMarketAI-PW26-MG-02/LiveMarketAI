from __future__ import annotations

from datetime import datetime

from ai_driven_retail_investor_confidence_meter.db.session import SessionLocal, init_db
from ai_driven_retail_investor_confidence_meter.repositories.sentiment_signal import sentiment_signal_repository as repo


def seed(n: int = 25) -> None:
    init_db()
    db = SessionLocal()
    try:
        for _ in range(n):
            repo.create(db, source="sample", symbol="sample", score=1.0, ts=datetime.utcnow())
        print(f"seeded {n} sentiment_signal rows")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
