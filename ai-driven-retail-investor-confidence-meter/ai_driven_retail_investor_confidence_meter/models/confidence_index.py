from __future__ import annotations

from datetime import datetime

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer,
                        JSON, String, Text)

from ..db.base import Base


class ConfidenceIndex(Base):
    __tablename__ = "confidence_indexs"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(255), nullable=True)
    value = Column(Float, nullable=True)
    ts = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ConfidenceIndex id={self.id}>"
