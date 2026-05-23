from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.liquidity_metric import liquidity_metric_repository as repo


def create_liquidity_metric(db: Session, **data):
    return repo.create(db, **data)


def list_liquidity_metric(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_liquidity_metric(db: Session) -> int:
    return repo.count(db)
