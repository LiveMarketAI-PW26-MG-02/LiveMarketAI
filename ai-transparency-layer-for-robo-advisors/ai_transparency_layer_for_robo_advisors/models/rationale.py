from __future__ import annotations

from datetime import datetime

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer,
                        JSON, String, Text)

from ..db.base import Base


class Rationale(Base):
    __tablename__ = "rationales"

    id = Column(Integer, primary_key=True, index=True)
    allocation_id = Column(Integer, nullable=True, index=True)
    text = Column(Text, nullable=True)
    factors = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Rationale id={self.id}>"
