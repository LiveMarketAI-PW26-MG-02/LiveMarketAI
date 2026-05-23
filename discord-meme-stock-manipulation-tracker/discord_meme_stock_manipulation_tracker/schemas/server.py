from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class ServerBase(BaseModel):
    name: Optional[str] = None
    members: Optional[int] = None


class ServerCreate(ServerBase):
    pass


class ServerRead(ServerBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
