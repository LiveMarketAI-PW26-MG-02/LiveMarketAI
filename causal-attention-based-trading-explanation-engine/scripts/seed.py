from __future__ import annotations

from datetime import datetime

from causal_attention_based_trading_explanation_engine.db.session import SessionLocal, init_db
from causal_attention_based_trading_explanation_engine.repositories.trade import trade_repository as repo


def seed(n: int = 25) -> None:
    init_db()
    db = SessionLocal()
    try:
        for _ in range(n):
            repo.create(db, symbol="sample", side="sample", size=1.0, ts=datetime.utcnow())
        print(f"seeded {n} trade rows")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
