from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class LiquidityMetricBase(BaseModel):
    institution: Optional[str] = None
    name: Optional[str] = None
    value: Optional[float] = None
    ts: Optional[datetime] = None


class LiquidityMetricCreate(LiquidityMetricBase):
    pass


class LiquidityMetricRead(LiquidityMetricBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
