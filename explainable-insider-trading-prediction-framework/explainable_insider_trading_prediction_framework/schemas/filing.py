from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class FilingBase(BaseModel):
    company: Optional[str] = None
    form_type: Optional[str] = None
    filed_at: Optional[datetime] = None
    material: Optional[bool] = None


class FilingCreate(FilingBase):
    pass


class FilingRead(FilingBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
