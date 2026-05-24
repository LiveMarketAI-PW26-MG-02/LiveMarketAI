from __future__ import annotations

from ..models.source import Source
from .base import CRUDRepository


class SourceRepository(CRUDRepository[Source]):
    def __init__(self) -> None:
        super().__init__(Source)


source_repository = SourceRepository()
