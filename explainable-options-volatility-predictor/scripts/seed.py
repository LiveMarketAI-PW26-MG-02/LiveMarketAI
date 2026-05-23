from __future__ import annotations

from datetime import datetime

from explainable_options_volatility_predictor.db.session import SessionLocal, init_db
from explainable_options_volatility_predictor.repositories.option import option_repository as repo


def seed(n: int = 25) -> None:
    init_db()
    db = SessionLocal()
    try:
        for _ in range(n):
            repo.create(db, underlying="sample", strike=1.0, expiry=datetime.utcnow(), kind="sample")
        print(f"seeded {n} option rows")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
