from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.server import server_repository as repo


def create_server(db: Session, **data):
    return repo.create(db, **data)


def list_server(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_server(db: Session) -> int:
    return repo.count(db)
