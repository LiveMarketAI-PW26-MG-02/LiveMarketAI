from __future__ import annotations

from datetime import datetime

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer,
                        JSON, String, Text)

from ..db.base import Base


class Decomposition(Base):
    __tablename__ = "decompositions"

    id = Column(Integer, primary_key=True, index=True)
    derivative_id = Column(Integer, nullable=True, index=True)
    factor = Column(String(255), nullable=True)
    contribution = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Decomposition id={self.id}>"
