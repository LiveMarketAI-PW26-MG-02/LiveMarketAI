from __future__ import annotations

from datetime import datetime

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer,
                        JSON, String, Text)

from ..db.base import Base


class MediaClip(Base):
    __tablename__ = "media_clips"

    id = Column(Integer, primary_key=True, index=True)
    source = Column(String(255), nullable=True)
    subject = Column(String(255), nullable=True)
    url = Column(String(255), nullable=True)
    ts = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<MediaClip id={self.id}>"
