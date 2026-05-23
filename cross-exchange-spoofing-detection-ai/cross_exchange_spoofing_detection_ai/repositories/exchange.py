from __future__ import annotations

from ..models.exchange import Exchange
from .base import CRUDRepository


class ExchangeRepository(CRUDRepository[Exchange]):
    def __init__(self) -> None:
        super().__init__(Exchange)


exchange_repository = ExchangeRepository()
