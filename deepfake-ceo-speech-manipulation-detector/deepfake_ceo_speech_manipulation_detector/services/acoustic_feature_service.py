from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.acoustic_feature import acoustic_feature_repository as repo


def create_acoustic_feature(db: Session, **data):
    return repo.create(db, **data)


def list_acoustic_feature(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_acoustic_feature(db: Session) -> int:
    return repo.count(db)
