from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.influencer import influencer_repository as repo


def create_influencer(db: Session, **data):
    return repo.create(db, **data)


def list_influencer(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_influencer(db: Session) -> int:
    return repo.count(db)
