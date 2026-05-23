from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.spoof_alert import spoof_alert_repository as repo


def create_spoof_alert(db: Session, **data):
    return repo.create(db, **data)


def list_spoof_alert(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_spoof_alert(db: Session) -> int:
    return repo.count(db)
