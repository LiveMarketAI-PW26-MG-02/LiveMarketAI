from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class CausalFactorBase(BaseModel):
    name: Optional[str] = None
    kept: Optional[bool] = None
    effect: Optional[float] = None


class CausalFactorCreate(CausalFactorBase):
    pass


class CausalFactorRead(CausalFactorBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
