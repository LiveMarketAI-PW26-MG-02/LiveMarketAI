from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.compliance_check import compliance_check_repository as repo


def create_compliance_check(db: Session, **data):
    return repo.create(db, **data)


def list_compliance_check(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_compliance_check(db: Session) -> int:
    return repo.count(db)
