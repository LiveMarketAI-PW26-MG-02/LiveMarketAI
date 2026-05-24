from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class PostBase(BaseModel):
    influencer: Optional[str] = None
    text: Optional[str] = None
    symbol: Optional[str] = None
    ts: Optional[datetime] = None


class PostCreate(PostBase):
    pass


class PostRead(PostBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
