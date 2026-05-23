from __future__ import annotations

from datetime import datetime

from explainable_liquidity_crisis_forecasting_engine.db.session import SessionLocal, init_db
from explainable_liquidity_crisis_forecasting_engine.repositories.liquidity_metric import liquidity_metric_repository as repo


def seed(n: int = 25) -> None:
    init_db()
    db = SessionLocal()
    try:
        for _ in range(n):
            repo.create(db, institution="sample", name="sample", value=1.0, ts=datetime.utcnow())
        print(f"seeded {n} liquidity_metric rows")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
