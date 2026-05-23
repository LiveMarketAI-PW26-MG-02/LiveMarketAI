from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class TransactionBase(BaseModel):
    tx_hash: Optional[str] = None
    src: Optional[str] = None
    dst: Optional[str] = None
    amount: Optional[float] = None
    ts: Optional[datetime] = None


class TransactionCreate(TransactionBase):
    pass


class TransactionRead(TransactionBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
