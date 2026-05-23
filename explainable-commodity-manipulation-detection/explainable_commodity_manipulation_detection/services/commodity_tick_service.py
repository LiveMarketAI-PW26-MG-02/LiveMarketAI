from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.commodity_tick import commodity_tick_repository as repo


def create_commodity_tick(db: Session, **data):
    return repo.create(db, **data)


def list_commodity_tick(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_commodity_tick(db: Session) -> int:
    return repo.count(db)
