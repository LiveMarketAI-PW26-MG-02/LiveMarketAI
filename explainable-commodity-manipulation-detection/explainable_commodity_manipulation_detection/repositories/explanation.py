from __future__ import annotations

from ..models.explanation import Explanation
from .base import CRUDRepository


class ExplanationRepository(CRUDRepository[Explanation]):
    def __init__(self) -> None:
        super().__init__(Explanation)


explanation_repository = ExplanationRepository()
