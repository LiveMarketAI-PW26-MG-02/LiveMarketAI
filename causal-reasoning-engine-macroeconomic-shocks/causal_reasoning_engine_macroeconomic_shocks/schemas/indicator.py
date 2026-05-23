from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class IndicatorBase(BaseModel):
    code: Optional[str] = None
    name: Optional[str] = None
    value: Optional[float] = None
    observed_at: Optional[datetime] = None


class IndicatorCreate(IndicatorBase):
    pass


class IndicatorRead(IndicatorBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
