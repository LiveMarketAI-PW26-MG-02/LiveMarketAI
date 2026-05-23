from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class DerivativeBase(BaseModel):
    symbol: Optional[str] = None
    kind: Optional[str] = None
    notional: Optional[float] = None
    expiry: Optional[datetime] = None


class DerivativeCreate(DerivativeBase):
    pass


class DerivativeRead(DerivativeBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
