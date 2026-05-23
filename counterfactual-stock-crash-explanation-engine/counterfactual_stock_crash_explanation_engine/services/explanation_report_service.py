from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.explanation_report import explanation_report_repository as repo


def create_explanation_report(db: Session, **data):
    return repo.create(db, **data)


def list_explanation_report(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_explanation_report(db: Session) -> int:
    return repo.count(db)
