from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class WalletBase(BaseModel):
    address: Optional[str] = None
    label: Optional[str] = None
    balance: Optional[float] = None
    first_seen: Optional[datetime] = None


class WalletCreate(WalletBase):
    pass


class WalletRead(WalletBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
