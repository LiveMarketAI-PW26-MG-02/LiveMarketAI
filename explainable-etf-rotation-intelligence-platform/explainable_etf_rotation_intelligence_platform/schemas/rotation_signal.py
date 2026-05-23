from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class RotationSignalBase(BaseModel):
    from_sector: Optional[str] = None
    to_sector: Optional[str] = None
    strength: Optional[float] = None
    ts: Optional[datetime] = None


class RotationSignalCreate(RotationSignalBase):
    pass


class RotationSignalRead(RotationSignalBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
