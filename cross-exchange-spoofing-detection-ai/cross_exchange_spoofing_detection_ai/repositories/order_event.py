from __future__ import annotations

from ..models.order_event import OrderEvent
from .base import CRUDRepository


class OrderEventRepository(CRUDRepository[OrderEvent]):
    def __init__(self) -> None:
        super().__init__(OrderEvent)


order_event_repository = OrderEventRepository()
