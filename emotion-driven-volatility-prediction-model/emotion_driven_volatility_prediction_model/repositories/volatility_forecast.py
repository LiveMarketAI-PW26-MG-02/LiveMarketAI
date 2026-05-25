from __future__ import annotations

from ..models.volatility_forecast import VolatilityForecast
from .base import CRUDRepository


class VolatilityForecastRepository(CRUDRepository[VolatilityForecast]):
    def __init__(self) -> None:
        super().__init__(VolatilityForecast)


volatility_forecast_repository = VolatilityForecastRepository()
