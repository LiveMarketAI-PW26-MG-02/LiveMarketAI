from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.insider import insider_repository as repo


def create_insider(db: Session, **data):
    return repo.create(db, **data)


def list_insider(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_insider(db: Session) -> int:
    return repo.count(db)
