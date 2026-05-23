from __future__ import annotations

from ..models.commodity_tick import CommodityTick
from .base import CRUDRepository


class CommodityTickRepository(CRUDRepository[CommodityTick]):
    def __init__(self) -> None:
        super().__init__(CommodityTick)


commodity_tick_repository = CommodityTickRepository()
