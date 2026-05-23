from __future__ import annotations

from datetime import datetime

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer,
                        JSON, String, Text)

from ..db.base import Base


class Company(Base):
    __tablename__ = "companys"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(255), nullable=True)
    name = Column(String(255), nullable=True)
    sector = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Company id={self.id}>"
