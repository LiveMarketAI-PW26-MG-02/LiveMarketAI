from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.scenario import scenario_repository as repo


def create_scenario(db: Session, **data):
    return repo.create(db, **data)


def list_scenario(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_scenario(db: Session) -> int:
    return repo.count(db)
