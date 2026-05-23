from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class InsiderBase(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    company: Optional[str] = None
    risk_tier: Optional[str] = None


class InsiderCreate(InsiderBase):
    pass


class InsiderRead(InsiderBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
