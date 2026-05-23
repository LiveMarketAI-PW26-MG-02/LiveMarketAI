from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.company import company_repository as repo


def create_company(db: Session, **data):
    return repo.create(db, **data)


def list_company(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_company(db: Session) -> int:
    return repo.count(db)
