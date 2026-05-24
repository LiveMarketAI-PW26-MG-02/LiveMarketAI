from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class WalletBase(BaseModel):
    address: Optional[str] = None
    balance: Optional[float] = None
    label: Optional[str] = None


class WalletCreate(WalletBase):
    pass


class WalletRead(WalletBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
