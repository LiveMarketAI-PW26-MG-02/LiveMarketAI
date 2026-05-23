from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.derivative import derivative_repository as repo


def create_derivative(db: Session, **data):
    return repo.create(db, **data)


def list_derivative(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_derivative(db: Session) -> int:
    return repo.count(db)
