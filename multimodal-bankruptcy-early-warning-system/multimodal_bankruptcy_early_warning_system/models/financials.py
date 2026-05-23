from __future__ import annotations

from datetime import datetime

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer,
                        JSON, String, Text)

from ..db.base import Base


class Financials(Base):
    __tablename__ = "financialss"

    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, nullable=True, index=True)
    period = Column(String(255), nullable=True)
    current_ratio = Column(Float, nullable=True)
    debt_to_equity = Column(Float, nullable=True)
    interest_coverage = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Financials id={self.id}>"
