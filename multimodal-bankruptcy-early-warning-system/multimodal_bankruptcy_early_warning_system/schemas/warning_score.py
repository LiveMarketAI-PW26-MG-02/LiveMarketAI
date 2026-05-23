from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class WarningScoreBase(BaseModel):
    company_id: Optional[int] = None
    score: Optional[float] = None
    lead_days: Optional[int] = None
    explanation: Optional[str] = None


class WarningScoreCreate(WarningScoreBase):
    pass


class WarningScoreRead(WarningScoreBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
