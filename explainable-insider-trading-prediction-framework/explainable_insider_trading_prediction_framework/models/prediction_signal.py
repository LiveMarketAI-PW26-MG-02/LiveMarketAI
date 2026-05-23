from __future__ import annotations

from datetime import datetime

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer,
                        JSON, String, Text)

from ..db.base import Base


class PredictionSignal(Base):
    __tablename__ = "prediction_signals"

    id = Column(Integer, primary_key=True, index=True)
    account = Column(String(255), nullable=True)
    score = Column(Float, nullable=True)
    window_start = Column(DateTime, nullable=True)
    features = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<PredictionSignal id={self.id}>"
