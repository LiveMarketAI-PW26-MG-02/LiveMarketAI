from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class DecompositionBase(BaseModel):
    derivative_id: Optional[int] = None
    factor: Optional[str] = None
    contribution: Optional[float] = None


class DecompositionCreate(DecompositionBase):
    pass


class DecompositionRead(DecompositionBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
