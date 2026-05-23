from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class UnderlyingBase(BaseModel):
    symbol: Optional[str] = None
    spot: Optional[float] = None
    realized_vol: Optional[float] = None


class UnderlyingCreate(UnderlyingBase):
    pass


class UnderlyingRead(UnderlyingBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
