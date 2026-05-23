from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class AllocationBase(BaseModel):
    etf_symbol: Optional[str] = None
    weight: Optional[float] = None


class AllocationCreate(AllocationBase):
    pass


class AllocationRead(AllocationBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
