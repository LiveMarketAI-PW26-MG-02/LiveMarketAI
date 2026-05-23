from __future__ import annotations

from datetime import datetime

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer,
                        JSON, String, Text)

from ..db.base import Base


class ManipulationAlert(Base):
    __tablename__ = "manipulation_alerts"

    id = Column(Integer, primary_key=True, index=True)
    instrument = Column(String(255), nullable=True)
    pattern = Column(String(255), nullable=True)
    confidence = Column(Float, nullable=True)
    raised_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ManipulationAlert id={self.id}>"
