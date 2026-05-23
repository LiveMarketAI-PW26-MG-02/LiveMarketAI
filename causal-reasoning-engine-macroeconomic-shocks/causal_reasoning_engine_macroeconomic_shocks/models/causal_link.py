from __future__ import annotations

from datetime import datetime

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer,
                        JSON, String, Text)

from ..db.base import Base


class CausalLink(Base):
    __tablename__ = "causal_links"

    id = Column(Integer, primary_key=True, index=True)
    cause = Column(String(255), nullable=True)
    effect = Column(String(255), nullable=True)
    strength = Column(Float, nullable=True)
    lag_days = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<CausalLink id={self.id}>"
