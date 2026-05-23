from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class GreeksBase(BaseModel):
    derivative_id: Optional[int] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None


class GreeksCreate(GreeksBase):
    pass


class GreeksRead(GreeksBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
