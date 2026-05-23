from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.pattern import pattern_repository as repo


def create_pattern(db: Session, **data):
    return repo.create(db, **data)


def list_pattern(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_pattern(db: Session) -> int:
    return repo.count(db)
