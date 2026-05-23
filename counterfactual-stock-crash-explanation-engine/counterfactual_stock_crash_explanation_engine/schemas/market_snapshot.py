from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class MarketSnapshotBase(BaseModel):
    symbol: Optional[str] = None
    price: Optional[float] = None
    volume: Optional[float] = None
    volatility: Optional[float] = None
    captured_at: Optional[datetime] = None


class MarketSnapshotCreate(MarketSnapshotBase):
    pass


class MarketSnapshotRead(MarketSnapshotBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
