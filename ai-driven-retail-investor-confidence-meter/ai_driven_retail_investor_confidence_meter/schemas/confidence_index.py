from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class ConfidenceIndexBase(BaseModel):
    symbol: Optional[str] = None
    value: Optional[float] = None
    ts: Optional[datetime] = None


class ConfidenceIndexCreate(ConfidenceIndexBase):
    pass


class ConfidenceIndexRead(ConfidenceIndexBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
