from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class PredictionSignalBase(BaseModel):
    account: Optional[str] = None
    score: Optional[float] = None
    window_start: Optional[datetime] = None
    features: Optional[dict] = None


class PredictionSignalCreate(PredictionSignalBase):
    pass


class PredictionSignalRead(PredictionSignalBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
