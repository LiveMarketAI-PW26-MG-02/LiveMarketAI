from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class AcousticFeatureBase(BaseModel):
    clip_id: Optional[int] = None
    name: Optional[str] = None
    value: Optional[float] = None


class AcousticFeatureCreate(AcousticFeatureBase):
    pass


class AcousticFeatureRead(AcousticFeatureBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
