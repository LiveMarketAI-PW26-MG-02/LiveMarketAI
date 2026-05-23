from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.confidence_index import confidence_index_repository as repo


def create_confidence_index(db: Session, **data):
    return repo.create(db, **data)


def list_confidence_index(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_confidence_index(db: Session) -> int:
    return repo.count(db)
