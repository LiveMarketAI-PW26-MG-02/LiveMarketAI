from __future__ import annotations

from datetime import datetime

from ceo_confidence_scoring_system.db.session import SessionLocal, init_db
from ceo_confidence_scoring_system.repositories.transcript import transcript_repository as repo


def seed(n: int = 25) -> None:
    init_db()
    db = SessionLocal()
    try:
        for _ in range(n):
            repo.create(db, company="sample", quarter="sample", text="sample", ts=datetime.utcnow())
        print(f"seeded {n} transcript rows")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
