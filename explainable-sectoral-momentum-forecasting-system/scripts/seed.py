from __future__ import annotations

from datetime import datetime

from explainable_sectoral_momentum_forecasting_system.db.session import SessionLocal, init_db
from explainable_sectoral_momentum_forecasting_system.repositories.sector import sector_repository as repo


def seed(n: int = 25) -> None:
    init_db()
    db = SessionLocal()
    try:
        for _ in range(n):
            repo.create(db, name="sample", code="sample")
        print(f"seeded {n} sector rows")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
