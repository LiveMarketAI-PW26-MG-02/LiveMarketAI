from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class RiskScoreBase(BaseModel):
    account: Optional[str] = None
    value: Optional[float] = None
    tier: Optional[str] = None
    explanation: Optional[str] = None


class RiskScoreCreate(RiskScoreBase):
    pass


class RiskScoreRead(RiskScoreBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
