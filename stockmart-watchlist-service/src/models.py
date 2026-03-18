from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import uuid

class WatchItem(BaseModel):
    symbol: str
    note: Optional[str] = None
    alert_above: Optional[float] = None
    alert_below: Optional[float] = None
    added_at: datetime = Field(default_factory=datetime.utcnow)

class Watchlist(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    name: str
    description: Optional[str] = None
    is_public: bool = False
    items: List[WatchItem] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class CreateWatchlistRequest(BaseModel):
    name: str
    description: Optional[str] = None
    is_public: bool = False
