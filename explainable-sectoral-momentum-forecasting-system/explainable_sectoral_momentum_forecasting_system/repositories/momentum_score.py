from __future__ import annotations

from ..models.momentum_score import MomentumScore
from .base import CRUDRepository


class MomentumScoreRepository(CRUDRepository[MomentumScore]):
    def __init__(self) -> None:
        super().__init__(MomentumScore)


momentum_score_repository = MomentumScoreRepository()
