from __future__ import annotations

from datetime import datetime

from explainable_etf_rotation_intelligence_platform.db.session import SessionLocal, init_db
from explainable_etf_rotation_intelligence_platform.repositories.e_t_f import e_t_f_repository as repo


def seed(n: int = 25) -> None:
    init_db()
    db = SessionLocal()
    try:
        for _ in range(n):
            repo.create(db, symbol="sample", name="sample", sector="sample")
        print(f"seeded {n} e_t_f rows")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
