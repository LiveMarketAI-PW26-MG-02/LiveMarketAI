from __future__ import annotations

from typing import Generic, Iterable, Optional, Type, TypeVar

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.base import Base

ModelT = TypeVar("ModelT", bound=Base)


class CRUDRepository(Generic[ModelT]):
    """Generic synchronous CRUD repository over a SQLAlchemy model."""

    def __init__(self, model: Type[ModelT]) -> None:
        self.model = model

    def create(self, db: Session, **data) -> ModelT:
        obj = self.model(**data)
        db.add(obj)
        db.commit()
        db.refresh(obj)
        return obj

    def bulk_create(self, db: Session, rows: Iterable[dict]) -> int:
        objs = [self.model(**r) for r in rows]
        db.add_all(objs)
        db.commit()
        return len(objs)

    def get(self, db: Session, obj_id: int) -> Optional[ModelT]:
        return db.get(self.model, obj_id)

    def list(self, db: Session, limit: int = 100, offset: int = 0) -> list[ModelT]:
        stmt = select(self.model).order_by(self.model.id.desc()).limit(limit).offset(offset)
        return list(db.execute(stmt).scalars().all())

    def update(self, db: Session, obj_id: int, **data) -> Optional[ModelT]:
        obj = self.get(db, obj_id)
        if obj is None:
            return None
        for k, v in data.items():
            if v is not None and hasattr(obj, k):
                setattr(obj, k, v)
        db.commit()
        db.refresh(obj)
        return obj

    def delete(self, db: Session, obj_id: int) -> bool:
        obj = self.get(db, obj_id)
        if obj is None:
            return False
        db.delete(obj)
        db.commit()
        return True

    def count(self, db: Session) -> int:
        return db.query(self.model).count()
