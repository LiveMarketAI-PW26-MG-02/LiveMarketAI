from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.stress_scenario import stress_scenario_repository as repo


def create_stress_scenario(db: Session, **data):
    return repo.create(db, **data)


def list_stress_scenario(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_stress_scenario(db: Session) -> int:
    return repo.count(db)
