from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.subject import subject_repository as repo


def create_subject(db: Session, **data):
    return repo.create(db, **data)


def list_subject(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_subject(db: Session) -> int:
    return repo.count(db)
