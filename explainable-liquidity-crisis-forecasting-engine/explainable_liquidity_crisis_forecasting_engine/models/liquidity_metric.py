from __future__ import annotations

from datetime import datetime

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer,
                        JSON, String, Text)

from ..db.base import Base


class LiquidityMetric(Base):
    __tablename__ = "liquidity_metrics"

    id = Column(Integer, primary_key=True, index=True)
    institution = Column(String(255), nullable=True)
    name = Column(String(255), nullable=True)
    value = Column(Float, nullable=True)
    ts = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<LiquidityMetric id={self.id}>"
