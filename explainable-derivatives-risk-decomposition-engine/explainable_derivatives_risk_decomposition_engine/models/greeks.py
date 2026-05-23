from __future__ import annotations

from datetime import datetime

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer,
                        JSON, String, Text)

from ..db.base import Base


class Greeks(Base):
    __tablename__ = "greekss"

    id = Column(Integer, primary_key=True, index=True)
    derivative_id = Column(Integer, nullable=True, index=True)
    delta = Column(Float, nullable=True)
    gamma = Column(Float, nullable=True)
    vega = Column(Float, nullable=True)
    theta = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Greeks id={self.id}>"
