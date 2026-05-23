from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class AttributionResultBase(BaseModel):
    claim_id: Optional[int] = None
    narrative: Optional[str] = None
    confidence: Optional[float] = None


class AttributionResultCreate(AttributionResultBase):
    pass


class AttributionResultRead(AttributionResultBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
