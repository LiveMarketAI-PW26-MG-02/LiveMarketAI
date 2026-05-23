from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.evidence import evidence_repository as repo


def create_evidence(db: Session, **data):
    return repo.create(db, **data)


def list_evidence(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_evidence(db: Session) -> int:
    return repo.count(db)
