from __future__ import annotations

from datetime import datetime

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer,
                        JSON, String, Text)

from ..db.base import Base


class AcousticFeature(Base):
    __tablename__ = "acoustic_features"

    id = Column(Integer, primary_key=True, index=True)
    clip_id = Column(Integer, nullable=True, index=True)
    name = Column(String(255), nullable=True)
    value = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<AcousticFeature id={self.id}>"
