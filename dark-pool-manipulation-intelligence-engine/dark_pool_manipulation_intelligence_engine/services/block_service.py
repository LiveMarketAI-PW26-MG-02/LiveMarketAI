from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.block import block_repository as repo


def create_block(db: Session, **data):
    return repo.create(db, **data)


def list_block(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_block(db: Session) -> int:
    return repo.count(db)
