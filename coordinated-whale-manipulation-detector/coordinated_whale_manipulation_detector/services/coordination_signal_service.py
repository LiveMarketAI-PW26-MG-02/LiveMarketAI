from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.coordination_signal import coordination_signal_repository as repo


def create_coordination_signal(db: Session, **data):
    return repo.create(db, **data)


def list_coordination_signal(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_coordination_signal(db: Session) -> int:
    return repo.count(db)
