from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.dark_pool_print import dark_pool_print_repository as repo


def create_dark_pool_print(db: Session, **data):
    return repo.create(db, **data)


def list_dark_pool_print(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_dark_pool_print(db: Session) -> int:
    return repo.count(db)
