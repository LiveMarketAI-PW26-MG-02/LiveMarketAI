from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.decomposition import decomposition_repository as repo


def create_decomposition(db: Session, **data):
    return repo.create(db, **data)


def list_decomposition(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_decomposition(db: Session) -> int:
    return repo.count(db)
