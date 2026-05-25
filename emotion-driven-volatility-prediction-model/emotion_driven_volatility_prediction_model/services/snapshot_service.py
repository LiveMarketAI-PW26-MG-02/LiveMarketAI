from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.snapshot import snapshot_repository as repo


def create_snapshot(db: Session, **data):
    return repo.create(db, **data)


def list_snapshot(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_snapshot(db: Session) -> int:
    return repo.count(db)
