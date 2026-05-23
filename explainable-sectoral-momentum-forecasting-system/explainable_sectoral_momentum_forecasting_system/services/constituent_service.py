from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.constituent import constituent_repository as repo


def create_constituent(db: Session, **data):
    return repo.create(db, **data)


def list_constituent(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_constituent(db: Session) -> int:
    return repo.count(db)
