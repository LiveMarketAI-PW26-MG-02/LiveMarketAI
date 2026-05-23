from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.financials import financials_repository as repo


def create_financials(db: Session, **data):
    return repo.create(db, **data)


def list_financials(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_financials(db: Session) -> int:
    return repo.count(db)
