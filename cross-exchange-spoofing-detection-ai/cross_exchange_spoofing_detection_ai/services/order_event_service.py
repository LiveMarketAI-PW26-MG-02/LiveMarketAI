from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.order_event import order_event_repository as repo


def create_order_event(db: Session, **data):
    return repo.create(db, **data)


def list_order_event(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_order_event(db: Session) -> int:
    return repo.count(db)
