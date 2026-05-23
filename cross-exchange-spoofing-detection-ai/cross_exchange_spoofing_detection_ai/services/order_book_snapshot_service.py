from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.order_book_snapshot import order_book_snapshot_repository as repo


def create_order_book_snapshot(db: Session, **data):
    return repo.create(db, **data)


def list_order_book_snapshot(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_order_book_snapshot(db: Session) -> int:
    return repo.count(db)
