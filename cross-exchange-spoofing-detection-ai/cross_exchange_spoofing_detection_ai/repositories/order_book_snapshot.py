from __future__ import annotations

from ..models.order_book_snapshot import OrderBookSnapshot
from .base import CRUDRepository


class OrderBookSnapshotRepository(CRUDRepository[OrderBookSnapshot]):
    def __init__(self) -> None:
        super().__init__(OrderBookSnapshot)


order_book_snapshot_repository = OrderBookSnapshotRepository()
