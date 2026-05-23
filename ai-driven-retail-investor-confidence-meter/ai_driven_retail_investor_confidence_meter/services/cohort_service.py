from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.cohort import cohort_repository as repo


def create_cohort(db: Session, **data):
    return repo.create(db, **data)


def list_cohort(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_cohort(db: Session) -> int:
    return repo.count(db)
