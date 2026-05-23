from __future__ import annotations

from datetime import datetime

from causal_reasoning_engine_macroeconomic_shocks.db.session import SessionLocal, init_db
from causal_reasoning_engine_macroeconomic_shocks.repositories.indicator import indicator_repository as repo


def seed(n: int = 25) -> None:
    init_db()
    db = SessionLocal()
    try:
        for _ in range(n):
            repo.create(db, code="sample", name="sample", value=1.0, observed_at=datetime.utcnow())
        print(f"seeded {n} indicator rows")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
