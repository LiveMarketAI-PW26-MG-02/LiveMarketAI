from __future__ import annotations

from ..models.confidence_score import ConfidenceScore
from .base import CRUDRepository


class ConfidenceScoreRepository(CRUDRepository[ConfidenceScore]):
    def __init__(self) -> None:
        super().__init__(ConfidenceScore)


confidence_score_repository = ConfidenceScoreRepository()
