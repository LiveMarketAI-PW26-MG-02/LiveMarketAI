from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class ClaimBase(BaseModel):
    text: Optional[str] = None
    symbol: Optional[str] = None
    label: Optional[str] = None
    ts: Optional[datetime] = None


class ClaimCreate(ClaimBase):
    pass


class ClaimRead(ClaimBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
