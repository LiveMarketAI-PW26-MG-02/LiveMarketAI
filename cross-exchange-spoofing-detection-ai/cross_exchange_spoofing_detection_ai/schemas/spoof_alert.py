from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class SpoofAlertBase(BaseModel):
    symbol: Optional[str] = None
    exchange: Optional[str] = None
    confidence: Optional[float] = None
    ts: Optional[datetime] = None


class SpoofAlertCreate(SpoofAlertBase):
    pass


class SpoofAlertRead(SpoofAlertBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
