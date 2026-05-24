from __future__ import annotations

from datetime import datetime

from coordinated_influencer_campaign_detector.db.session import SessionLocal, init_db
from coordinated_influencer_campaign_detector.repositories.influencer import influencer_repository as repo


def seed(n: int = 25) -> None:
    init_db()
    db = SessionLocal()
    try:
        for _ in range(n):
            repo.create(db, handle="sample", platform="sample", followers=1)
        print(f"seeded {n} influencer rows")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
