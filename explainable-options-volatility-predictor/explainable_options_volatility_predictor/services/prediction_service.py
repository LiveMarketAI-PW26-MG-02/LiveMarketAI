from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.prediction import prediction_repository as repo


def create_prediction(db: Session, **data):
    return repo.create(db, **data)


def list_prediction(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_prediction(db: Session) -> int:
    return repo.count(db)
