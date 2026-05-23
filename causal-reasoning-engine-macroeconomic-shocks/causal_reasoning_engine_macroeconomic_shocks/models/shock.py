from __future__ import annotations

from datetime import datetime

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer,
                        JSON, String, Text)

from ..db.base import Base


class Shock(Base):
    __tablename__ = "shocks"

    id = Column(Integer, primary_key=True, index=True)
    indicator = Column(String(255), nullable=True)
    magnitude = Column(Float, nullable=True)
    description = Column(Text, nullable=True)
    occurred_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Shock id={self.id}>"
