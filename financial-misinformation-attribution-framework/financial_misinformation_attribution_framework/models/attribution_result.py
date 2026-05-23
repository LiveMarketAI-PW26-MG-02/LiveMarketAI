from __future__ import annotations

from datetime import datetime

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer,
                        JSON, String, Text)

from ..db.base import Base


class AttributionResult(Base):
    __tablename__ = "attribution_results"

    id = Column(Integer, primary_key=True, index=True)
    claim_id = Column(Integer, nullable=True, index=True)
    narrative = Column(String(255), nullable=True)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<AttributionResult id={self.id}>"
