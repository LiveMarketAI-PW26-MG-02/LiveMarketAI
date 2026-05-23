from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.token import token_repository as repo


def create_token(db: Session, **data):
    return repo.create(db, **data)


def list_token(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_token(db: Session) -> int:
    return repo.count(db)
