from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class FinancialsBase(BaseModel):
    company_id: Optional[int] = None
    period: Optional[str] = None
    current_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    interest_coverage: Optional[float] = None


class FinancialsCreate(FinancialsBase):
    pass


class FinancialsRead(FinancialsBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
