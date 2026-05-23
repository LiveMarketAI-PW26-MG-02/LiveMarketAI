from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.forecast import forecast_repository as repo


def create_forecast(db: Session, **data):
    return repo.create(db, **data)


def list_forecast(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_forecast(db: Session) -> int:
    return repo.count(db)
