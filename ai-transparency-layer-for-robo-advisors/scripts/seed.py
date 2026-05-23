from __future__ import annotations

from datetime import datetime

from ai_transparency_layer_for_robo_advisors.db.session import SessionLocal, init_db
from ai_transparency_layer_for_robo_advisors.repositories.advisor import advisor_repository as repo


def seed(n: int = 25) -> None:
    init_db()
    db = SessionLocal()
    try:
        for _ in range(n):
            repo.create(db, name="sample", strategy="sample", active=False)
        print(f"seeded {n} advisor rows")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
