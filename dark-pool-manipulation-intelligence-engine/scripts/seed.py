from __future__ import annotations

from datetime import datetime

from dark_pool_manipulation_intelligence_engine.db.session import SessionLocal, init_db
from dark_pool_manipulation_intelligence_engine.repositories.dark_pool_print import dark_pool_print_repository as repo


def seed(n: int = 25) -> None:
    init_db()
    db = SessionLocal()
    try:
        for _ in range(n):
            repo.create(db, symbol="sample", size=1.0, price=1.0, ts=datetime.utcnow())
        print(f"seeded {n} dark_pool_print rows")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
