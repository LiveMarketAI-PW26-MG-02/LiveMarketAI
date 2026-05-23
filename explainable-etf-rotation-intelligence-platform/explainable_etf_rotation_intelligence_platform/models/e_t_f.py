from __future__ import annotations

from datetime import datetime

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer,
                        JSON, String, Text)

from ..db.base import Base


class ETF(Base):
    __tablename__ = "e_t_fs"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(255), nullable=True)
    name = Column(String(255), nullable=True)
    sector = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ETF id={self.id}>"
