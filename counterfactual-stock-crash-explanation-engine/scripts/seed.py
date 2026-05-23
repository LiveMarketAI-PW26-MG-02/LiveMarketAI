from __future__ import annotations

from datetime import datetime

from counterfactual_stock_crash_explanation_engine.db.session import SessionLocal, init_db
from counterfactual_stock_crash_explanation_engine.repositories.crash_event import crash_event_repository as repo


def seed(n: int = 25) -> None:
    init_db()
    db = SessionLocal()
    try:
        for _ in range(n):
            repo.create(db, symbol="sample", drawdown_pct=1.0, started_at=datetime.utcnow(), severity="sample", resolved=False)
        print(f"seeded {n} crash_event rows")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
