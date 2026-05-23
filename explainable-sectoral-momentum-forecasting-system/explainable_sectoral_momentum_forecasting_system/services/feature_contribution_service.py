from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.feature_contribution import feature_contribution_repository as repo


def create_feature_contribution(db: Session, **data):
    return repo.create(db, **data)


def list_feature_contribution(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_feature_contribution(db: Session) -> int:
    return repo.count(db)
