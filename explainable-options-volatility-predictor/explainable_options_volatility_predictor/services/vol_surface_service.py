from __future__ import annotations

from sqlalchemy.orm import Session

from ..repositories.vol_surface import vol_surface_repository as repo


def create_vol_surface(db: Session, **data):
    return repo.create(db, **data)


def list_vol_surface(db: Session, limit: int = 100):
    return repo.list(db, limit=limit)


def count_vol_surface(db: Session) -> int:
    return repo.count(db)
