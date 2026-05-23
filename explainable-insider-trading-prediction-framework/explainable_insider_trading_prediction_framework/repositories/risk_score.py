from __future__ import annotations

from ..models.risk_score import RiskScore
from .base import CRUDRepository


class RiskScoreRepository(CRUDRepository[RiskScore]):
    def __init__(self) -> None:
        super().__init__(RiskScore)


risk_score_repository = RiskScoreRepository()
