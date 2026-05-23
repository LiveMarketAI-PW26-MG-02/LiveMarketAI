from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.attribution_result import attribution_result_repository as repo


def create_attribution_result(db: Session, **data):
    return repo.create(db, **data)


def list_attribution_result(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_attribution_result(db: Session) -> int:
    return repo.count(db)
