from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.whale_cluster import whale_cluster_repository as repo


def create_whale_cluster(db: Session, **data):
    return repo.create(db, **data)


def list_whale_cluster(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_whale_cluster(db: Session) -> int:
    return repo.count(db)
