from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class CompanyBase(BaseModel):
    ticker: Optional[str] = None
    name: Optional[str] = None
    sector: Optional[str] = None


class CompanyCreate(CompanyBase):
    pass


class CompanyRead(CompanyBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
