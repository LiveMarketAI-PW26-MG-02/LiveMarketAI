from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.attention_weight import attention_weight_repository as repo


def create_attention_weight(db: Session, **data):
    return repo.create(db, **data)


def list_attention_weight(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_attention_weight(db: Session) -> int:
    return repo.count(db)
