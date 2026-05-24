from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.edge import edge_repository as repo


def create_edge(db: Session, **data):
    return repo.create(db, **data)


def list_edge(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_edge(db: Session) -> int:
    return repo.count(db)
