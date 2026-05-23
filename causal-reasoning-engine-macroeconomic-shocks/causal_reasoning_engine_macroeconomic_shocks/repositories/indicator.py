from __future__ import annotations

from ..models.indicator import Indicator
from .base import CRUDRepository


class IndicatorRepository(CRUDRepository[Indicator]):
    def __init__(self) -> None:
        super().__init__(Indicator)


indicator_repository = IndicatorRepository()
