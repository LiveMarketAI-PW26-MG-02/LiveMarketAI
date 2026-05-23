from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class InfluenceEdgeBase(BaseModel):
    src: Optional[str] = None
    dst: Optional[str] = None
    weight: Optional[float] = None


class InfluenceEdgeCreate(InfluenceEdgeBase):
    pass


class InfluenceEdgeRead(InfluenceEdgeBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
