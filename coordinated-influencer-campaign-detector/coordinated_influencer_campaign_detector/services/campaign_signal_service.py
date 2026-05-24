from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.campaign_signal import campaign_signal_repository as repo


def create_campaign_signal(db: Session, **data):
    return repo.create(db, **data)


def list_campaign_signal(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_campaign_signal(db: Session) -> int:
    return repo.count(db)
