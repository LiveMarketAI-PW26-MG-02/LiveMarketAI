from __future__ import annotations

from datetime import datetime

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer,
                        JSON, String, Text)

from ..db.base import Base


class Constituent(Base):
    __tablename__ = "constituents"

    id = Column(Integer, primary_key=True, index=True)
    sector = Column(String(255), nullable=True)
    symbol = Column(String(255), nullable=True)
    weight = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Constituent id={self.id}>"
