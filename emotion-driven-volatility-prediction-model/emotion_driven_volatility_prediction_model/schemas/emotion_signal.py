from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class EmotionSignalBase(BaseModel):
    symbol: Optional[str] = None
    emotion: Optional[str] = None
    score: Optional[float] = None
    ts: Optional[datetime] = None


class EmotionSignalCreate(EmotionSignalBase):
    pass


class EmotionSignalRead(EmotionSignalBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
