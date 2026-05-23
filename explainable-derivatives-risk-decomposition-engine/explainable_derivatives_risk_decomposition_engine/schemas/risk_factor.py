from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class RiskFactorBase(BaseModel):
    name: Optional[str] = None
    bucket: Optional[str] = None
    exposure: Optional[float] = None


class RiskFactorCreate(RiskFactorBase):
    pass


class RiskFactorRead(RiskFactorBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
