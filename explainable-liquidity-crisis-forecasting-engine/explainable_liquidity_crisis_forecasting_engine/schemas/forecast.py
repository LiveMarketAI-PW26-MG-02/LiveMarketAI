from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class ForecastBase(BaseModel):
    metric: Optional[str] = None
    value: Optional[float] = None
    lower: Optional[float] = None
    upper: Optional[float] = None


class ForecastCreate(ForecastBase):
    pass


class ForecastRead(ForecastBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
