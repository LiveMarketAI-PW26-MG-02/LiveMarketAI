from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class CrashEventBase(BaseModel):
    symbol: Optional[str] = None
    drawdown_pct: Optional[float] = None
    started_at: Optional[datetime] = None
    severity: Optional[str] = None
    resolved: Optional[bool] = None


class CrashEventCreate(CrashEventBase):
    pass


class CrashEventRead(CrashEventBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
