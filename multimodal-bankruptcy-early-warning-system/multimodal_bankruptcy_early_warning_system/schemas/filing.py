from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class FilingBase(BaseModel):
    company_id: Optional[int] = None
    form: Optional[str] = None
    filed_at: Optional[datetime] = None


class FilingCreate(FilingBase):
    pass


class FilingRead(FilingBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
