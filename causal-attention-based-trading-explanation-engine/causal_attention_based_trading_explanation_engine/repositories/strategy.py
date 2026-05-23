from __future__ import annotations

from ..models.strategy import Strategy
from .base import CRUDRepository


class StrategyRepository(CRUDRepository[Strategy]):
    def __init__(self) -> None:
        super().__init__(Strategy)


strategy_repository = StrategyRepository()
