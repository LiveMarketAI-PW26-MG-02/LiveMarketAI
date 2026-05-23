from __future__ import annotations

from datetime import datetime

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer,
                        JSON, String, Text)

from ..db.base import Base


class StressScenario(Base):
    __tablename__ = "stress_scenarios"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=True)
    shock = Column(JSON, nullable=True)
    pnl = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<StressScenario id={self.id}>"
