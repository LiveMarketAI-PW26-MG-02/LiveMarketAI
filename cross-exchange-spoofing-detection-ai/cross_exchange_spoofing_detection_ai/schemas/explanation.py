from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class ExplanationBase(BaseModel):
    alert_id: Optional[int] = None
    summary: Optional[str] = None
    attributions: Optional[dict] = None


class ExplanationCreate(ExplanationBase):
    pass


class ExplanationRead(ExplanationBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
