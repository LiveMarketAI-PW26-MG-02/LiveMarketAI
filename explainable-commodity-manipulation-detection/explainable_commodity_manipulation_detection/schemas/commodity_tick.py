from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class CommodityTickBase(BaseModel):
    instrument: Optional[str] = None
    venue: Optional[str] = None
    price: Optional[float] = None
    volume: Optional[float] = None
    ts: Optional[datetime] = None


class CommodityTickCreate(CommodityTickBase):
    pass


class CommodityTickRead(CommodityTickBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
