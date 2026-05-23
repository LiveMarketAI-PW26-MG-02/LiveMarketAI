from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.crash_event import crash_event_repository as repo


def create_crash_event(db: Session, **data):
    return repo.create(db, **data)


def list_crash_event(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_crash_event(db: Session) -> int:
    return repo.count(db)
