from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.option import option_repository as repo


def create_option(db: Session, **data):
    return repo.create(db, **data)


def list_option(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_option(db: Session) -> int:
    return repo.count(db)
