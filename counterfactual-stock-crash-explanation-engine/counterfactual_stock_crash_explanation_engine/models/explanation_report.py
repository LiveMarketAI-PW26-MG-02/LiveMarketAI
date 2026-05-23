from __future__ import annotations

from datetime import datetime

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer,
                        JSON, String, Text)

from ..db.base import Base


class ExplanationReport(Base):
    __tablename__ = "explanation_reports"

    id = Column(Integer, primary_key=True, index=True)
    crash_event_id = Column(Integer, nullable=True, index=True)
    summary = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    payload = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ExplanationReport id={self.id}>"
