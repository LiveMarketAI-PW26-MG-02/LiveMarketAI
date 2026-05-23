from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class SnapshotBase(BaseModel):
    symbol: Optional[str] = None
    payload: Optional[dict] = None
    ts: Optional[datetime] = None


class SnapshotCreate(SnapshotBase):
    pass


class SnapshotRead(SnapshotBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
