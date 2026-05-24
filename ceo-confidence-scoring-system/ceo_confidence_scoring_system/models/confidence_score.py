from __future__ import annotations

from datetime import datetime

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer,
                        JSON, String, Text)

from ..db.base import Base


class ConfidenceScore(Base):
    __tablename__ = "confidence_scores"

    id = Column(Integer, primary_key=True, index=True)
    transcript_id = Column(Integer, nullable=True, index=True)
    value = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ConfidenceScore id={self.id}>"
