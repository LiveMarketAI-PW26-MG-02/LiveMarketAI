from __future__ import annotations

from datetime import datetime

from emotion_driven_volatility_prediction_model.db.session import SessionLocal, init_db
from emotion_driven_volatility_prediction_model.repositories.emotion_signal import emotion_signal_repository as repo


def seed(n: int = 25) -> None:
    init_db()
    db = SessionLocal()
    try:
        for _ in range(n):
            repo.create(db, symbol="sample", emotion="sample", score=1.0, ts=datetime.utcnow())
        print(f"seeded {n} emotion_signal rows")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
