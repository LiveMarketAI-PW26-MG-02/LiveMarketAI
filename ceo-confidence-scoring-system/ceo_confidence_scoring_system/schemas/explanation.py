from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class ExplanationBase(BaseModel):
    score_id: Optional[int] = None
    summary: Optional[str] = None
    breakdown: Optional[dict] = None


class ExplanationCreate(ExplanationBase):
    pass


class ExplanationRead(ExplanationBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
