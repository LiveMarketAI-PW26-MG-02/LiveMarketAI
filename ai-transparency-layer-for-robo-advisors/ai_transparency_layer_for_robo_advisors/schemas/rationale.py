from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class RationaleBase(BaseModel):
    allocation_id: Optional[int] = None
    text: Optional[str] = None
    factors: Optional[dict] = None


class RationaleCreate(RationaleBase):
    pass


class RationaleRead(RationaleBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
