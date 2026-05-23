from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class OptionBase(BaseModel):
    underlying: Optional[str] = None
    strike: Optional[float] = None
    expiry: Optional[datetime] = None
    kind: Optional[str] = None


class OptionCreate(OptionBase):
    pass


class OptionRead(OptionBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
