from __future__ import annotations

from datetime import datetime

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer,
                        JSON, String, Text)

from ..db.base import Base


class CrisisSignal(Base):
    __tablename__ = "crisis_signals"

    id = Column(Integer, primary_key=True, index=True)
    score = Column(Float, nullable=True)
    horizon_days = Column(Integer, nullable=True)
    raised_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<CrisisSignal id={self.id}>"
