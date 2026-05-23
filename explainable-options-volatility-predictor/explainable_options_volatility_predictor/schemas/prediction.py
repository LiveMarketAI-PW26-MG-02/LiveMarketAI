from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class PredictionBase(BaseModel):
    option_id: Optional[int] = None
    iv_forecast: Optional[float] = None
    horizon_days: Optional[int] = None


class PredictionCreate(PredictionBase):
    pass


class PredictionRead(PredictionBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
