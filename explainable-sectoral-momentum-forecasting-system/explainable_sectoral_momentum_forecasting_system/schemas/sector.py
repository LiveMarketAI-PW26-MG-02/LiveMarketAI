from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class SectorBase(BaseModel):
    name: Optional[str] = None
    code: Optional[str] = None


class SectorCreate(SectorBase):
    pass


class SectorRead(SectorBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
