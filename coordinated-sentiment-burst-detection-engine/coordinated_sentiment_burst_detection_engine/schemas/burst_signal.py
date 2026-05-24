from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class BurstSignalBase(BaseModel):
    symbol: Optional[str] = None
    intensity: Optional[float] = None
    ts: Optional[datetime] = None


class BurstSignalCreate(BurstSignalBase):
    pass


class BurstSignalRead(BurstSignalBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
