from __future__ import annotations

from datetime import datetime

from financial_misinformation_attribution_framework.db.session import SessionLocal, init_db
from financial_misinformation_attribution_framework.repositories.claim import claim_repository as repo


def seed(n: int = 25) -> None:
    init_db()
    db = SessionLocal()
    try:
        for _ in range(n):
            repo.create(db, text="sample", symbol="sample", label="sample", ts=datetime.utcnow())
        print(f"seeded {n} claim rows")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
