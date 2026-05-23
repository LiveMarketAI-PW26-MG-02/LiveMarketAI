from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class MomentumScoreBase(BaseModel):
    sector: Optional[str] = None
    value: Optional[float] = None
    ts: Optional[datetime] = None


class MomentumScoreCreate(MomentumScoreBase):
    pass


class MomentumScoreRead(MomentumScoreBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
