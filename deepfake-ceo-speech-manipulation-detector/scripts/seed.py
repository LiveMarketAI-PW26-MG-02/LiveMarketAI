from __future__ import annotations

from datetime import datetime

from deepfake_ceo_speech_manipulation_detector.db.session import SessionLocal, init_db
from deepfake_ceo_speech_manipulation_detector.repositories.media_clip import media_clip_repository as repo


def seed(n: int = 25) -> None:
    init_db()
    db = SessionLocal()
    try:
        for _ in range(n):
            repo.create(db, source="sample", subject="sample", url="sample", ts=datetime.utcnow())
        print(f"seeded {n} media_clip rows")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
