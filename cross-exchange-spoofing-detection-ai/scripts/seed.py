from __future__ import annotations

from datetime import datetime

from cross_exchange_spoofing_detection_ai.db.session import SessionLocal, init_db
from cross_exchange_spoofing_detection_ai.repositories.order_book_snapshot import order_book_snapshot_repository as repo


def seed(n: int = 25) -> None:
    init_db()
    db = SessionLocal()
    try:
        for _ in range(n):
            repo.create(db, exchange="sample", symbol="sample", payload={}, ts=datetime.utcnow())
        print(f"seeded {n} order_book_snapshot rows")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
