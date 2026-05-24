from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.source import source_repository as repo


def create_source(db: Session, **data):
    return repo.create(db, **data)


def list_source(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_source(db: Session) -> int:
    return repo.count(db)
