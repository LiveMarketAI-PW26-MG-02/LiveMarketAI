from __future__ import annotations

from datetime import datetime

from explainable_derivatives_risk_decomposition_engine.db.session import SessionLocal, init_db
from explainable_derivatives_risk_decomposition_engine.repositories.derivative import derivative_repository as repo


def seed(n: int = 25) -> None:
    init_db()
    db = SessionLocal()
    try:
        for _ in range(n):
            repo.create(db, symbol="sample", kind="sample", notional=1.0, expiry=datetime.utcnow())
        print(f"seeded {n} derivative rows")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
