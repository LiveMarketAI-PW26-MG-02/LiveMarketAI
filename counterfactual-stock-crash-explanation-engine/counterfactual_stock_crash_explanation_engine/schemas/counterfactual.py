from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class CounterfactualBase(BaseModel):
    crash_event_id: Optional[int] = None
    flipped: Optional[bool] = None
    delta: Optional[dict] = None
    distance: Optional[float] = None
    narrative: Optional[str] = None


class CounterfactualCreate(CounterfactualBase):
    pass


class CounterfactualRead(CounterfactualBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
