from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class ExplanationReportBase(BaseModel):
    crash_event_id: Optional[int] = None
    summary: Optional[str] = None
    confidence: Optional[float] = None
    payload: Optional[dict] = None


class ExplanationReportCreate(ExplanationReportBase):
    pass


class ExplanationReportRead(ExplanationReportBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
