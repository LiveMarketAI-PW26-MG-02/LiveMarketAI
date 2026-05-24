from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.post import post_repository as repo


def create_post(db: Session, **data):
    return repo.create(db, **data)


def list_post(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_post(db: Session) -> int:
    return repo.count(db)
