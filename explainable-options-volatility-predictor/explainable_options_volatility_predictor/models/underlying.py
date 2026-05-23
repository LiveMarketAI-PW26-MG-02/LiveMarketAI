from __future__ import annotations

from datetime import datetime

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer,
                        JSON, String, Text)

from ..db.base import Base


class Underlying(Base):
    __tablename__ = "underlyings"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(255), nullable=True)
    spot = Column(Float, nullable=True)
    realized_vol = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Underlying id={self.id}>"
