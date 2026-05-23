from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class TokenBase(BaseModel):
    symbol: Optional[str] = None
    chain: Optional[str] = None
    market_cap: Optional[float] = None


class TokenCreate(TokenBase):
    pass


class TokenRead(TokenBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
