from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.allocation import allocation_repository as repo


def create_allocation(db: Session, **data):
    return repo.create(db, **data)


def list_allocation(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_allocation(db: Session) -> int:
    return repo.count(db)
