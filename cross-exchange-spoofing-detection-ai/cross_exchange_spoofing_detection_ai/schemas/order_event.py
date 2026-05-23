from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class OrderEventBase(BaseModel):
    exchange: Optional[str] = None
    symbol: Optional[str] = None
    action: Optional[str] = None
    size: Optional[float] = None
    ts: Optional[datetime] = None


class OrderEventCreate(OrderEventBase):
    pass


class OrderEventRead(OrderEventBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
