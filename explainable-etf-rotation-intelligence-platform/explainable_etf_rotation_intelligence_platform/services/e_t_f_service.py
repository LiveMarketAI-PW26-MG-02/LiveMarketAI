from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.e_t_f import e_t_f_repository as repo


def create_e_t_f(db: Session, **data):
    return repo.create(db, **data)


def list_e_t_f(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_e_t_f(db: Session) -> int:
    return repo.count(db)
