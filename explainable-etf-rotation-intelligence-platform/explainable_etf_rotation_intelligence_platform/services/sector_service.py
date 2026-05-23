from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.sector import sector_repository as repo


def create_sector(db: Session, **data):
    return repo.create(db, **data)


def list_sector(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_sector(db: Session) -> int:
    return repo.count(db)
