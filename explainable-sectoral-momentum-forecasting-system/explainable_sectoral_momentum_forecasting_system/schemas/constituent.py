from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class ConstituentBase(BaseModel):
    sector: Optional[str] = None
    symbol: Optional[str] = None
    weight: Optional[float] = None


class ConstituentCreate(ConstituentBase):
    pass


class ConstituentRead(ConstituentBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
