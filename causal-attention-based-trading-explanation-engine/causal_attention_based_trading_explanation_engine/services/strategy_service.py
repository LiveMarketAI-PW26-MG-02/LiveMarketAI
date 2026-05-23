from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.strategy import strategy_repository as repo


def create_strategy(db: Session, **data):
    return repo.create(db, **data)


def list_strategy(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_strategy(db: Session) -> int:
    return repo.count(db)
