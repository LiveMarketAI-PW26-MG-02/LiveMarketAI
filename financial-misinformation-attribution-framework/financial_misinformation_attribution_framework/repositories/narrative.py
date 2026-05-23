from __future__ import annotations

from ..models.narrative import Narrative
from .base import CRUDRepository


class NarrativeRepository(CRUDRepository[Narrative]):
    def __init__(self) -> None:
        super().__init__(Narrative)


narrative_repository = NarrativeRepository()
