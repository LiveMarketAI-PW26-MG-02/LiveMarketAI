from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class VenueBase(BaseModel):
    name: Optional[str] = None
    dark: Optional[bool] = None


class VenueCreate(VenueBase):
    pass


class VenueRead(VenueBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
