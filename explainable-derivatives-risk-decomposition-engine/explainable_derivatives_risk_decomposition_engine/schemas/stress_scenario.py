from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class StressScenarioBase(BaseModel):
    name: Optional[str] = None
    shock: Optional[dict] = None
    pnl: Optional[float] = None


class StressScenarioCreate(StressScenarioBase):
    pass


class StressScenarioRead(StressScenarioBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
