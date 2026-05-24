from __future__ import annotations

from datetime import datetime

from coordinated_sentiment_burst_detection_engine.db.session import SessionLocal, init_db
from coordinated_sentiment_burst_detection_engine.repositories.post import post_repository as repo


def seed(n: int = 25) -> None:
    init_db()
    db = SessionLocal()
    try:
        for _ in range(n):
            repo.create(db, source="sample", text="sample", symbol="sample", ts=datetime.utcnow())
        print(f"seeded {n} post rows")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
