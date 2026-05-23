from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class StrategyBase(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    live: Optional[bool] = None


class StrategyCreate(StrategyBase):
    pass


class StrategyRead(StrategyBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
