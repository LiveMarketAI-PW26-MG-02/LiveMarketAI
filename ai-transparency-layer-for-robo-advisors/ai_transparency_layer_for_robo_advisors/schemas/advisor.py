from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class AdvisorBase(BaseModel):
    name: Optional[str] = None
    strategy: Optional[str] = None
    active: Optional[bool] = None


class AdvisorCreate(AdvisorBase):
    pass


class AdvisorRead(AdvisorBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
