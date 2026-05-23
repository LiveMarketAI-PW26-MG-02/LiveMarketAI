from __future__ import annotations

from datetime import datetime

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer,
                        JSON, String, Text)

from ..db.base import Base


class Forecast(Base):
    __tablename__ = "forecasts"

    id = Column(Integer, primary_key=True, index=True)
    indicator = Column(String(255), nullable=True)
    horizon_days = Column(Integer, nullable=True)
    value = Column(Float, nullable=True)
    lower = Column(Float, nullable=True)
    upper = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Forecast id={self.id}>"
