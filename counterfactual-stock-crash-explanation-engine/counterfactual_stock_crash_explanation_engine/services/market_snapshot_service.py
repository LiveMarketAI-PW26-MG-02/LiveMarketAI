from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.market_snapshot import market_snapshot_repository as repo


def create_market_snapshot(db: Session, **data):
    return repo.create(db, **data)


def list_market_snapshot(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_market_snapshot(db: Session) -> int:
    return repo.count(db)
