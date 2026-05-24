from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class WhaleGroupBase(BaseModel):
    name: Optional[str] = None
    size: Optional[int] = None
    centrality: Optional[float] = None


class WhaleGroupCreate(WhaleGroupBase):
    pass


class WhaleGroupRead(WhaleGroupBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
