from __future__ import annotations

from ..models.liquidity_metric import LiquidityMetric
from .base import CRUDRepository


class LiquidityMetricRepository(CRUDRepository[LiquidityMetric]):
    def __init__(self) -> None:
        super().__init__(LiquidityMetric)


liquidity_metric_repository = LiquidityMetricRepository()
