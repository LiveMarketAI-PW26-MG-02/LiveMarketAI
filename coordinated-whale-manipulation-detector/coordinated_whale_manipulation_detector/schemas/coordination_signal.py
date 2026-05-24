from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class CoordinationSignalBase(BaseModel):
    score: Optional[float] = None
    cohort: Optional[str] = None
    ts: Optional[datetime] = None


class CoordinationSignalCreate(CoordinationSignalBase):
    pass


class CoordinationSignalRead(CoordinationSignalBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
