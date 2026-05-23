from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.indicator import indicator_repository as repo


def create_indicator(db: Session, **data):
    return repo.create(db, **data)


def list_indicator(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_indicator(db: Session) -> int:
    return repo.count(db)
