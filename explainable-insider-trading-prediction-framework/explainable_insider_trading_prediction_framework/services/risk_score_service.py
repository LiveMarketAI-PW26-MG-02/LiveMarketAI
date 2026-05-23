from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.risk_score import risk_score_repository as repo


def create_risk_score(db: Session, **data):
    return repo.create(db, **data)


def list_risk_score(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_risk_score(db: Session) -> int:
    return repo.count(db)
