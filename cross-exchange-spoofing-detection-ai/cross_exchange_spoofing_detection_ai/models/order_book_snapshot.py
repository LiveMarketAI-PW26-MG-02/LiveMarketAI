from __future__ import annotations

from datetime import datetime

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer,
                        JSON, String, Text)

from ..db.base import Base


class OrderBookSnapshot(Base):
    __tablename__ = "order_book_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    exchange = Column(String(255), nullable=True)
    symbol = Column(String(255), nullable=True)
    payload = Column(JSON, nullable=True)
    ts = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<OrderBookSnapshot id={self.id}>"
