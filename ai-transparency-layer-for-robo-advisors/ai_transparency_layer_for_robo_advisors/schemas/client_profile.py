from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class ClientProfileBase(BaseModel):
    client_ref: Optional[str] = None
    risk: Optional[str] = None
    horizon: Optional[str] = None


class ClientProfileCreate(ClientProfileBase):
    pass


class ClientProfileRead(ClientProfileBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
