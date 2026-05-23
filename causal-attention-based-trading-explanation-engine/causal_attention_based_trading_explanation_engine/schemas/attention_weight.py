from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class AttentionWeightBase(BaseModel):
    trade_id: Optional[int] = None
    feature: Optional[str] = None
    weight: Optional[float] = None


class AttentionWeightCreate(AttentionWeightBase):
    pass


class AttentionWeightRead(AttentionWeightBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
