from __future__ import annotations

from datetime import datetime

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer,
                        JSON, String, Text)

from ..db.base import Base


class CommodityTick(Base):
    __tablename__ = "commodity_ticks"

    id = Column(Integer, primary_key=True, index=True)
    instrument = Column(String(255), nullable=True)
    venue = Column(String(255), nullable=True)
    price = Column(Float, nullable=True)
    volume = Column(Float, nullable=True)
    ts = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<CommodityTick id={self.id}>"
