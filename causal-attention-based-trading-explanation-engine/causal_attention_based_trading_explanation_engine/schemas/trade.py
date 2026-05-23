from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class TradeBase(BaseModel):
    symbol: Optional[str] = None
    side: Optional[str] = None
    size: Optional[float] = None
    ts: Optional[datetime] = None


class TradeCreate(TradeBase):
    pass


class TradeRead(TradeBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
