from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class ComplianceCheckBase(BaseModel):
    allocation_id: Optional[int] = None
    passed: Optional[bool] = None
    notes: Optional[str] = None


class ComplianceCheckCreate(ComplianceCheckBase):
    pass


class ComplianceCheckRead(ComplianceCheckBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
