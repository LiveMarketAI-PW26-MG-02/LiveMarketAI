from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.advisor import advisor_repository as repo


def create_advisor(db: Session, **data):
    return repo.create(db, **data)


def list_advisor(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_advisor(db: Session) -> int:
    return repo.count(db)
