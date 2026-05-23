from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class OrderBookSnapshotBase(BaseModel):
    exchange: Optional[str] = None
    symbol: Optional[str] = None
    payload: Optional[dict] = None
    ts: Optional[datetime] = None


class OrderBookSnapshotCreate(OrderBookSnapshotBase):
    pass


class OrderBookSnapshotRead(OrderBookSnapshotBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
