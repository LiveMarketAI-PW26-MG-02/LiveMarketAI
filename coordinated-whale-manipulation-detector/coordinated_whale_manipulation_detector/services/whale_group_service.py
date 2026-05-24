from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.whale_group import whale_group_repository as repo


def create_whale_group(db: Session, **data):
    return repo.create(db, **data)


def list_whale_group(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_whale_group(db: Session) -> int:
    return repo.count(db)
