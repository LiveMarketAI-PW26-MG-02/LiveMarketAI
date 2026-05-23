from __future__ import annotations

from datetime import datetime

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer,
                        JSON, String, Text)

from ..db.base import Base


class AttentionWeight(Base):
    __tablename__ = "attention_weights"

    id = Column(Integer, primary_key=True, index=True)
    trade_id = Column(Integer, nullable=True, index=True)
    feature = Column(String(255), nullable=True)
    weight = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<AttentionWeight id={self.id}>"
