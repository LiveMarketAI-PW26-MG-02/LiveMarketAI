from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class DetectionAlertBase(BaseModel):
    signal_id: Optional[int] = None
    summary: Optional[str] = None
    evidence: Optional[dict] = None


class DetectionAlertCreate(DetectionAlertBase):
    pass


class DetectionAlertRead(DetectionAlertBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
