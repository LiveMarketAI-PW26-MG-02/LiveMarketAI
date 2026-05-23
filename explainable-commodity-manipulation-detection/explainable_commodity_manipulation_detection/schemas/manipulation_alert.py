from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class ManipulationAlertBase(BaseModel):
    instrument: Optional[str] = None
    pattern: Optional[str] = None
    confidence: Optional[float] = None
    raised_at: Optional[datetime] = None


class ManipulationAlertCreate(ManipulationAlertBase):
    pass


class ManipulationAlertRead(ManipulationAlertBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
