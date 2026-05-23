from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class NarrativeBase(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class NarrativeCreate(NarrativeBase):
    pass


class NarrativeRead(NarrativeBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
