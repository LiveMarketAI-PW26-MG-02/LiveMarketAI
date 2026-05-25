from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class VolatilityForecastBase(BaseModel):
    symbol: Optional[str] = None
    value: Optional[float] = None
    horizon_days: Optional[int] = None


class VolatilityForecastCreate(VolatilityForecastBase):
    pass


class VolatilityForecastRead(VolatilityForecastBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
