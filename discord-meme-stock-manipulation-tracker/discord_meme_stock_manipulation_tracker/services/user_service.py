from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.user import user_repository as repo


def create_user(db: Session, **data):
    return repo.create(db, **data)


def list_user(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_user(db: Session) -> int:
    return repo.count(db)
