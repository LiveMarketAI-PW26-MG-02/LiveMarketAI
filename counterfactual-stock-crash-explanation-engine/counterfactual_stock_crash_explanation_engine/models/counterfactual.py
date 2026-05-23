from __future__ import annotations

from datetime import datetime

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer,
                        JSON, String, Text)

from ..db.base import Base


class Counterfactual(Base):
    __tablename__ = "counterfactuals"

    id = Column(Integer, primary_key=True, index=True)
    crash_event_id = Column(Integer, nullable=True, index=True)
    flipped = Column(Boolean, nullable=True, default=False)
    delta = Column(JSON, nullable=True)
    distance = Column(Float, nullable=True)
    narrative = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Counterfactual id={self.id}>"
