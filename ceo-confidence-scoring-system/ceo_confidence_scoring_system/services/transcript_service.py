from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.transcript import transcript_repository as repo


def create_transcript(db: Session, **data):
    return repo.create(db, **data)


def list_transcript(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_transcript(db: Session) -> int:
    return repo.count(db)
