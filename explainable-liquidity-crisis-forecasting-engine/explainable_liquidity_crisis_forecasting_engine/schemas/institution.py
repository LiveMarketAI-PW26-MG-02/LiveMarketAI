from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class InstitutionBase(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    systemic: Optional[bool] = None


class InstitutionCreate(InstitutionBase):
    pass


class InstitutionRead(InstitutionBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
