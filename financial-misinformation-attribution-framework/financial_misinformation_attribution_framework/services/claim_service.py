from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.claim import claim_repository as repo


def create_claim(db: Session, **data):
    return repo.create(db, **data)


def list_claim(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_claim(db: Session) -> int:
    return repo.count(db)
