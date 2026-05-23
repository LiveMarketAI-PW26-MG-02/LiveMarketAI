from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.narrative import narrative_repository as repo


def create_narrative(db: Session, **data):
    return repo.create(db, **data)


def list_narrative(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_narrative(db: Session) -> int:
    return repo.count(db)
