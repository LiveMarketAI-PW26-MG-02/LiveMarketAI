from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class TranscriptBase(BaseModel):
    company: Optional[str] = None
    quarter: Optional[str] = None
    text: Optional[str] = None
    ts: Optional[datetime] = None


class TranscriptCreate(TranscriptBase):
    pass


class TranscriptRead(TranscriptBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
