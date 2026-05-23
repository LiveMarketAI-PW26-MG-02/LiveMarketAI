from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.causal_factor import causal_factor_repository as repo


def create_causal_factor(db: Session, **data):
    return repo.create(db, **data)


def list_causal_factor(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_causal_factor(db: Session) -> int:
    return repo.count(db)
