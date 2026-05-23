from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.shock import shock_repository as repo


def create_shock(db: Session, **data):
    return repo.create(db, **data)


def list_shock(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_shock(db: Session) -> int:
    return repo.count(db)
