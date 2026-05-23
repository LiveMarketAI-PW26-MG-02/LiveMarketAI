from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.venue import venue_repository as repo


def create_venue(db: Session, **data):
    return repo.create(db, **data)


def list_venue(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_venue(db: Session) -> int:
    return repo.count(db)
