from __future__ import annotations

from ..models.risk_factor import RiskFactor
from .base import CRUDRepository


class RiskFactorRepository(CRUDRepository[RiskFactor]):
    def __init__(self) -> None:
        super().__init__(RiskFactor)


risk_factor_repository = RiskFactorRepository()
