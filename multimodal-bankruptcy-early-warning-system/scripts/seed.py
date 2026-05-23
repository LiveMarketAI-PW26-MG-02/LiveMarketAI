from __future__ import annotations

from datetime import datetime

from multimodal_bankruptcy_early_warning_system.db.session import SessionLocal, init_db
from multimodal_bankruptcy_early_warning_system.repositories.company import company_repository as repo


def seed(n: int = 25) -> None:
    init_db()
    db = SessionLocal()
    try:
        for _ in range(n):
            repo.create(db, ticker="sample", name="sample", sector="sample")
        print(f"seeded {n} company rows")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
