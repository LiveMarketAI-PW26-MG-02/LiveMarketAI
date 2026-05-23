from __future__ import annotations

from datetime import datetime

from discord_meme_stock_manipulation_tracker.db.session import SessionLocal, init_db
from discord_meme_stock_manipulation_tracker.repositories.server import server_repository as repo


def seed(n: int = 25) -> None:
    init_db()
    db = SessionLocal()
    try:
        for _ in range(n):
            repo.create(db, name="sample", members=1)
        print(f"seeded {n} server rows")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
