from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class CampaignSignalBase(BaseModel):
    symbol: Optional[str] = None
    score: Optional[float] = None
    ts: Optional[datetime] = None


class CampaignSignalCreate(CampaignSignalBase):
    pass


class CampaignSignalRead(CampaignSignalBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
