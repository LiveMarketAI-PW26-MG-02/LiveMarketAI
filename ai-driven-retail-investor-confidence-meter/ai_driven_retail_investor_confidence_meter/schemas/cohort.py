from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class CohortBase(BaseModel):
    name: Optional[str] = None
    size: Optional[int] = None
    region: Optional[str] = None


class CohortCreate(CohortBase):
    pass


class CohortRead(CohortBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
