from __future__ import annotations

from ..models.pattern import Pattern
from .base import CRUDRepository


class PatternRepository(CRUDRepository[Pattern]):
    def __init__(self) -> None:
        super().__init__(Pattern)


pattern_repository = PatternRepository()
