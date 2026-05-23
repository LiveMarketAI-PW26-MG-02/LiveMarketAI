from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class PatternBase(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    severity: Optional[str] = None


class PatternCreate(PatternBase):
    pass


class PatternRead(PatternBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
