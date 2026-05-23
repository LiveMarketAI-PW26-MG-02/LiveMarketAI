from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.filing import filing_repository as repo


def create_filing(db: Session, **data):
    return repo.create(db, **data)


def list_filing(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_filing(db: Session) -> int:
    return repo.count(db)
