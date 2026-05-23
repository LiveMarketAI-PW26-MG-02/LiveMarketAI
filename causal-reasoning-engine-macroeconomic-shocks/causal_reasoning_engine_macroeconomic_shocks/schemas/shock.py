from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class ShockBase(BaseModel):
    indicator: Optional[str] = None
    magnitude: Optional[float] = None
    description: Optional[str] = None
    occurred_at: Optional[datetime] = None


class ShockCreate(ShockBase):
    pass


class ShockRead(ShockBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
