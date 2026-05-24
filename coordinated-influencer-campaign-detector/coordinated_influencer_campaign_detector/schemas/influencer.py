from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class InfluencerBase(BaseModel):
    handle: Optional[str] = None
    platform: Optional[str] = None
    followers: Optional[int] = None


class InfluencerCreate(InfluencerBase):
    pass


class InfluencerRead(InfluencerBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
