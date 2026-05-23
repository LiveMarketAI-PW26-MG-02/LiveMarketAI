from __future__ import annotations

from datetime import datetime

from explainable_commodity_manipulation_detection.db.session import SessionLocal, init_db
from explainable_commodity_manipulation_detection.repositories.commodity_tick import commodity_tick_repository as repo


def seed(n: int = 25) -> None:
    init_db()
    db = SessionLocal()
    try:
        for _ in range(n):
            repo.create(db, instrument="sample", venue="sample", price=1.0, volume=1.0, ts=datetime.utcnow())
        print(f"seeded {n} commodity_tick rows")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
