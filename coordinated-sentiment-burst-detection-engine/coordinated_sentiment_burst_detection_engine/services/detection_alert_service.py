from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.detection_alert import detection_alert_repository as repo


def create_detection_alert(db: Session, **data):
    return repo.create(db, **data)


def list_detection_alert(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_detection_alert(db: Session) -> int:
    return repo.count(db)
