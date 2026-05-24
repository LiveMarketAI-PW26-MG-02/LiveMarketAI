from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class ConfidenceScoreBase(BaseModel):
    transcript_id: Optional[int] = None
    value: Optional[float] = None


class ConfidenceScoreCreate(ConfidenceScoreBase):
    pass


class ConfidenceScoreRead(ConfidenceScoreBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
