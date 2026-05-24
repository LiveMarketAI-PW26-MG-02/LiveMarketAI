from __future__ import annotations

from datetime import datetime

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer,
                        JSON, String, Text)

from ..db.base import Base


class WhaleGroup(Base):
    __tablename__ = "whale_groups"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=True)
    size = Column(Integer, nullable=True)
    centrality = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<WhaleGroup id={self.id}>"
