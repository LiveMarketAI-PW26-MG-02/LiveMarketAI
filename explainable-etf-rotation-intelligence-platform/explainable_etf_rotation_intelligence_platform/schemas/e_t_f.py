from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class ETFBase(BaseModel):
    symbol: Optional[str] = None
    name: Optional[str] = None
    sector: Optional[str] = None


class ETFCreate(ETFBase):
    pass


class ETFRead(ETFBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
