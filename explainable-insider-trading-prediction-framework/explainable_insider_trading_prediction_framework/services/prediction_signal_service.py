from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.prediction_signal import prediction_signal_repository as repo


def create_prediction_signal(db: Session, **data):
    return repo.create(db, **data)


def list_prediction_signal(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_prediction_signal(db: Session) -> int:
    return repo.count(db)
