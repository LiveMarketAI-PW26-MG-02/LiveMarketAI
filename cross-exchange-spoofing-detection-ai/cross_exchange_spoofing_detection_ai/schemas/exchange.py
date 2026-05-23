from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class ExchangeBase(BaseModel):
    name: Optional[str] = None
    region: Optional[str] = None


class ExchangeCreate(ExchangeBase):
    pass


class ExchangeRead(ExchangeBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
