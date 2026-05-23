from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class SourceBase(BaseModel):
    name: Optional[str] = None
    platform: Optional[str] = None
    reliability: Optional[float] = None


class SourceCreate(SourceBase):
    pass


class SourceRead(SourceBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
