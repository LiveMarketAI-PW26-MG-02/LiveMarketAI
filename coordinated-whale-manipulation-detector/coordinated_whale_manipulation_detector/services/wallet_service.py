from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.wallet import wallet_repository as repo


def create_wallet(db: Session, **data):
    return repo.create(db, **data)


def list_wallet(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_wallet(db: Session) -> int:
    return repo.count(db)
