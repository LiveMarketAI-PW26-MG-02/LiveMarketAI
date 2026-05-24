from __future__ import annotations

from datetime import datetime

from coordinated_whale_manipulation_detector.db.session import SessionLocal, init_db
from coordinated_whale_manipulation_detector.repositories.wallet import wallet_repository as repo


def seed(n: int = 25) -> None:
    init_db()
    db = SessionLocal()
    try:
        for _ in range(n):
            repo.create(db, address="sample", balance=1.0, label="sample")
        print(f"seeded {n} wallet rows")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
