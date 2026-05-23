from __future__ import annotations

from ..models.trade import Trade
from .base import CRUDRepository


class TradeRepository(CRUDRepository[Trade]):
    def __init__(self) -> None:
        super().__init__(Trade)


trade_repository = TradeRepository()
