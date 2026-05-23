from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.manipulation_alert import manipulation_alert_repository as repo


def create_manipulation_alert(db: Session, **data):
    return repo.create(db, **data)


def list_manipulation_alert(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_manipulation_alert(db: Session) -> int:
    return repo.count(db)
