from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.news_signal import news_signal_repository as repo


def create_news_signal(db: Session, **data):
    return repo.create(db, **data)


def list_news_signal(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_news_signal(db: Session) -> int:
    return repo.count(db)
