from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.exchange import exchange_repository as repo


def create_exchange(db: Session, **data):
    return repo.create(db, **data)


def list_exchange(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_exchange(db: Session) -> int:
    return repo.count(db)
