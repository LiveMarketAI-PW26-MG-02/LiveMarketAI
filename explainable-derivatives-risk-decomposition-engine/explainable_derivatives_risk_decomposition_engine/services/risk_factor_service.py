from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.risk_factor import risk_factor_repository as repo


def create_risk_factor(db: Session, **data):
    return repo.create(db, **data)


def list_risk_factor(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_risk_factor(db: Session) -> int:
    return repo.count(db)
