from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class BlockBase(BaseModel):
    symbol: Optional[str] = None
    size: Optional[float] = None
    counterparty: Optional[str] = None
    ts: Optional[datetime] = None


class BlockCreate(BlockBase):
    pass


class BlockRead(BlockBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
