from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class ScenarioBase(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    perturbation: Optional[dict] = None
    crash_probability: Optional[float] = None


class ScenarioCreate(ScenarioBase):
    pass


class ScenarioRead(ScenarioBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
