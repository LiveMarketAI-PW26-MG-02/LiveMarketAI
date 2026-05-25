from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class MediaClipBase(BaseModel):
    source: Optional[str] = None
    subject: Optional[str] = None
    url: Optional[str] = None
    ts: Optional[datetime] = None


class MediaClipCreate(MediaClipBase):
    pass


class MediaClipRead(MediaClipBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
