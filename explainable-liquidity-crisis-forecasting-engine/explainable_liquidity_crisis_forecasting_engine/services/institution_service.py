from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.institution import institution_repository as repo


def create_institution(db: Session, **data):
    return repo.create(db, **data)


def list_institution(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_institution(db: Session) -> int:
    return repo.count(db)
