from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class DarkPoolPrintBase(BaseModel):
    symbol: Optional[str] = None
    size: Optional[float] = None
    price: Optional[float] = None
    ts: Optional[datetime] = None


class DarkPoolPrintCreate(DarkPoolPrintBase):
    pass


class DarkPoolPrintRead(DarkPoolPrintBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
