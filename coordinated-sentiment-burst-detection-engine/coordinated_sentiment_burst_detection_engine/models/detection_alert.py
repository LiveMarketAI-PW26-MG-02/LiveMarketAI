from __future__ import annotations

from datetime import datetime

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer,
                        JSON, String, Text)

from ..db.base import Base


class DetectionAlert(Base):
    __tablename__ = "detection_alerts"

    id = Column(Integer, primary_key=True, index=True)
    burst_id = Column(Integer, nullable=True, index=True)
    summary = Column(Text, nullable=True)
    evidence = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<DetectionAlert id={self.id}>"
