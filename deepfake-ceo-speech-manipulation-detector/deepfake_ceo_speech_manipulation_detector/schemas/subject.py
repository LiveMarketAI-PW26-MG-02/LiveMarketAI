from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class SubjectBase(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    company: Optional[str] = None


class SubjectCreate(SubjectBase):
    pass


class SubjectRead(SubjectBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
