from __future__ import annotations

from datetime import datetime

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer,
                        JSON, String, Text)

from ..db.base import Base


class Filing(Base):
    __tablename__ = "filings"

    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, nullable=True, index=True)
    form = Column(String(255), nullable=True)
    filed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Filing id={self.id}>"
