from __future__ import annotations

from ..models.forecast import Forecast
from .base import CRUDRepository


class ForecastRepository(CRUDRepository[Forecast]):
    def __init__(self) -> None:
        super().__init__(Forecast)


forecast_repository = ForecastRepository()
