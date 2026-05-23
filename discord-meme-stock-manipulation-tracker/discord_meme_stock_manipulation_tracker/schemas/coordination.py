from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class CoordinationBase(BaseModel):
    symbol: Optional[str] = None
    score: Optional[float] = None
    evidence: Optional[dict] = None


class CoordinationCreate(CoordinationBase):
    pass


class CoordinationRead(CoordinationBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
