from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.executive import executive_repository as repo


def create_executive(db: Session, **data):
    return repo.create(db, **data)


def list_executive(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_executive(db: Session) -> int:
    return repo.count(db)
