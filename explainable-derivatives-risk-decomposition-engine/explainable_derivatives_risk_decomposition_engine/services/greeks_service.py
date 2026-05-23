from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.greeks import greeks_repository as repo


def create_greeks(db: Session, **data):
    return repo.create(db, **data)


def list_greeks(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_greeks(db: Session) -> int:
    return repo.count(db)
