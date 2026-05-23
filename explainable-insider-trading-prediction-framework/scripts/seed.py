from __future__ import annotations

from datetime import datetime

from explainable_insider_trading_prediction_framework.db.session import SessionLocal, init_db
from explainable_insider_trading_prediction_framework.repositories.trade import trade_repository as repo


def seed(n: int = 25) -> None:
    init_db()
    db = SessionLocal()
    try:
        for _ in range(n):
            repo.create(db, symbol="sample", account="sample", size=1.0, price=1.0, executed_at=datetime.utcnow())
        print(f"seeded {n} trade rows")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
