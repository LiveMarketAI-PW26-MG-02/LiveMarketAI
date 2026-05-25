from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.media_clip import media_clip_repository as repo


def create_media_clip(db: Session, **data):
    return repo.create(db, **data)


def list_media_clip(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_media_clip(db: Session) -> int:
    return repo.count(db)
