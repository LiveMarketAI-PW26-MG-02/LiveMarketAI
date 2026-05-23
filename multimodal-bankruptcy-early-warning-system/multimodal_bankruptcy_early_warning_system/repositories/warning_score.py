from __future__ import annotations

from ..models.warning_score import WarningScore
from .base import CRUDRepository


class WarningScoreRepository(CRUDRepository[WarningScore]):
    def __init__(self) -> None:
        super().__init__(WarningScore)


warning_score_repository = WarningScoreRepository()
