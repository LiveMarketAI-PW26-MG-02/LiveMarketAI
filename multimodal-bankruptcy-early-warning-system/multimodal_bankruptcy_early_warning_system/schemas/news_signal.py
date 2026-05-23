from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class NewsSignalBase(BaseModel):
    company_id: Optional[int] = None
    sentiment: Optional[float] = None
    source: Optional[str] = None
    ts: Optional[datetime] = None


class NewsSignalCreate(NewsSignalBase):
    pass


class NewsSignalRead(NewsSignalBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
