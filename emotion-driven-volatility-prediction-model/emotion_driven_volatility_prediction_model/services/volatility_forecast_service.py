from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.volatility_forecast import volatility_forecast_repository as repo


def create_volatility_forecast(db: Session, **data):
    return repo.create(db, **data)


def list_volatility_forecast(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_volatility_forecast(db: Session) -> int:
    return repo.count(db)
