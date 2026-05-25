from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class AuthenticitySignalBase(BaseModel):
    clip_id: Optional[int] = None
    fake_probability: Optional[float] = None
    ts: Optional[datetime] = None


class AuthenticitySignalCreate(AuthenticitySignalBase):
    pass


class AuthenticitySignalRead(AuthenticitySignalBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
