from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class ExecutiveBase(BaseModel):
    name: Optional[str] = None
    company: Optional[str] = None
    role: Optional[str] = None


class ExecutiveCreate(ExecutiveBase):
    pass


class ExecutiveRead(ExecutiveBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
