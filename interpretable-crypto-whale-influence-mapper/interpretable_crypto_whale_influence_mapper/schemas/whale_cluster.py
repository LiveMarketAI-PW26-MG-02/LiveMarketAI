from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class WhaleClusterBase(BaseModel):
    name: Optional[str] = None
    size: Optional[int] = None
    centrality: Optional[float] = None


class WhaleClusterCreate(WhaleClusterBase):
    pass


class WhaleClusterRead(WhaleClusterBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
