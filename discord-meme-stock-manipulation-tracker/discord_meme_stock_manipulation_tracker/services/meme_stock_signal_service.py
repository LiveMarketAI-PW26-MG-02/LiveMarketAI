from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.meme_stock_signal import meme_stock_signal_repository as repo


def create_meme_stock_signal(db: Session, **data):
    return repo.create(db, **data)


def list_meme_stock_signal(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_meme_stock_signal(db: Session) -> int:
    return repo.count(db)
