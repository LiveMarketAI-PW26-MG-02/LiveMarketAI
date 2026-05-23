from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.transaction import transaction_repository as repo


def create_transaction(db: Session, **data):
    return repo.create(db, **data)


def list_transaction(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_transaction(db: Session) -> int:
    return repo.count(db)
