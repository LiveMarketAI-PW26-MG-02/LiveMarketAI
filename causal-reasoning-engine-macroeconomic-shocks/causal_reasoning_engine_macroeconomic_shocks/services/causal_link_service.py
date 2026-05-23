from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.causal_link import causal_link_repository as repo


def create_causal_link(db: Session, **data):
    return repo.create(db, **data)


def list_causal_link(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_causal_link(db: Session) -> int:
    return repo.count(db)
