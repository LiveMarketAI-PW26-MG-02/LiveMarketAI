from __future__ import annotations

from ..models.market_snapshot import MarketSnapshot
from .base import CRUDRepository


class MarketSnapshotRepository(CRUDRepository[MarketSnapshot]):
    def __init__(self) -> None:
        super().__init__(MarketSnapshot)


market_snapshot_repository = MarketSnapshotRepository()
