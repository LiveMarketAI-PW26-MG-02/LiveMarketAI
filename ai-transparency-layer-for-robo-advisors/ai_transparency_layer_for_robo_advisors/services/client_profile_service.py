from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.client_profile import client_profile_repository as repo


def create_client_profile(db: Session, **data):
    return repo.create(db, **data)


def list_client_profile(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_client_profile(db: Session) -> int:
    return repo.count(db)
