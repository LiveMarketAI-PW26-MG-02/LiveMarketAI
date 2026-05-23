from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.counterfactual import counterfactual_repository as repo


def create_counterfactual(db: Session, **data):
    return repo.create(db, **data)


def list_counterfactual(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_counterfactual(db: Session) -> int:
    return repo.count(db)
