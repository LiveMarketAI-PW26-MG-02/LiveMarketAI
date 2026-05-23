from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class SentimentSignalBase(BaseModel):
    source: Optional[str] = None
    symbol: Optional[str] = None
    score: Optional[float] = None
    ts: Optional[datetime] = None


class SentimentSignalCreate(SentimentSignalBase):
    pass


class SentimentSignalRead(SentimentSignalBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
