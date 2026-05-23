from __future__ import annotations

from ..models.confidence_index import ConfidenceIndex
from .base import CRUDRepository


class ConfidenceIndexRepository(CRUDRepository[ConfidenceIndex]):
    def __init__(self) -> None:
        super().__init__(ConfidenceIndex)


confidence_index_repository = ConfidenceIndexRepository()
