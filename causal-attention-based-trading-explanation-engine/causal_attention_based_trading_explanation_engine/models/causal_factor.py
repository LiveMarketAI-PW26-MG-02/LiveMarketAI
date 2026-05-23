from __future__ import annotations

from datetime import datetime

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer,
                        JSON, String, Text)

from ..db.base import Base


class CausalFactor(Base):
    __tablename__ = "causal_factors"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=True)
    kept = Column(Boolean, nullable=True, default=False)
    effect = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<CausalFactor id={self.id}>"
