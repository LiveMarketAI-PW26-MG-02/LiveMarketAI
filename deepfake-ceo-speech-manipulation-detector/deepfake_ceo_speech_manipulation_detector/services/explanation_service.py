from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.explanation import explanation_repository as repo


def create_explanation(db: Session, **data):
    return repo.create(db, **data)


def list_explanation(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_explanation(db: Session) -> int:
    return repo.count(db)
