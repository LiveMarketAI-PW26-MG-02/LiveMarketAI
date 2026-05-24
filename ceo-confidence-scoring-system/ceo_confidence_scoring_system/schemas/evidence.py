from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class EvidenceBase(BaseModel):
    transcript_id: Optional[int] = None
    kind: Optional[str] = None
    weight: Optional[float] = None


class EvidenceCreate(EvidenceBase):
    pass


class EvidenceRead(EvidenceBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
