from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class ClusterBase(BaseModel):
    name: Optional[str] = None
    size: Optional[int] = None


class ClusterCreate(ClusterBase):
    pass


class ClusterRead(ClusterBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
