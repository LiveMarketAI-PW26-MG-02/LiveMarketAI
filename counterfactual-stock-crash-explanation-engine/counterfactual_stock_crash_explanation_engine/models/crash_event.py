from __future__ import annotations

from datetime import datetime

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer,
                        JSON, String, Text)

from ..db.base import Base


class CrashEvent(Base):
    __tablename__ = "crash_events"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(255), nullable=True)
    drawdown_pct = Column(Float, nullable=True)
    started_at = Column(DateTime, nullable=True)
    severity = Column(String(255), nullable=True)
    resolved = Column(Boolean, nullable=True, default=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<CrashEvent id={self.id}>"
