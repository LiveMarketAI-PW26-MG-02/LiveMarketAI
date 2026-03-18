from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional
from datetime import datetime
import uuid

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"

class CartItem(BaseModel):
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float = Field(..., gt=0)
    limit_price: Optional[float] = None
    added_at: datetime = Field(default_factory=datetime.utcnow)

class Cart(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    items: list[CartItem] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    checked_out: bool = False

class CheckoutResult(BaseModel):
    cart_id: str
    orders_placed: int
    total_items: int
    status: str
    placed_at: datetime = Field(default_factory=datetime.utcnow)
