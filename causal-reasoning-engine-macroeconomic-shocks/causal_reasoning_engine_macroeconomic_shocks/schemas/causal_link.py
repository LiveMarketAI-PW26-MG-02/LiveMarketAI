from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class CausalLinkBase(BaseModel):
    cause: Optional[str] = None
    effect: Optional[str] = None
    strength: Optional[float] = None
    lag_days: Optional[int] = None


class CausalLinkCreate(CausalLinkBase):
    pass


class CausalLinkRead(CausalLinkBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
