from __future__ import annotations

from ..models.meme_stock_signal import MemeStockSignal
from .base import CRUDRepository


class MemeStockSignalRepository(CRUDRepository[MemeStockSignal]):
    def __init__(self) -> None:
        super().__init__(MemeStockSignal)


meme_stock_signal_repository = MemeStockSignalRepository()
