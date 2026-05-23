from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class VolSurfaceBase(BaseModel):
    underlying: Optional[str] = None
    snapshot: Optional[dict] = None
    ts: Optional[datetime] = None


class VolSurfaceCreate(VolSurfaceBase):
    pass


class VolSurfaceRead(VolSurfaceBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
