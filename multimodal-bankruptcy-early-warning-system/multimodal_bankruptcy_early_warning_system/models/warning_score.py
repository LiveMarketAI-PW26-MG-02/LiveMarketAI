from __future__ import annotations

from datetime import datetime

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer,
                        JSON, String, Text)

from ..db.base import Base


class WarningScore(Base):
    __tablename__ = "warning_scores"

    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, nullable=True, index=True)
    score = Column(Float, nullable=True)
    lead_days = Column(Integer, nullable=True)
    explanation = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<WarningScore id={self.id}>"
