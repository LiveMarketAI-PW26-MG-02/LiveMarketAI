from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class CrisisSignalBase(BaseModel):
    score: Optional[float] = None
    horizon_days: Optional[int] = None
    raised_at: Optional[datetime] = None


class CrisisSignalCreate(CrisisSignalBase):
    pass


class CrisisSignalRead(CrisisSignalBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
