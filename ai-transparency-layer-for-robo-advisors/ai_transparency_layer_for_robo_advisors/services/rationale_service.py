from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.rationale import rationale_repository as repo


def create_rationale(db: Session, **data):
    return repo.create(db, **data)


def list_rationale(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_rationale(db: Session) -> int:
    return repo.count(db)
