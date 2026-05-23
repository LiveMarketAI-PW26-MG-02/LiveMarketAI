from __future__ import annotations

from ..models.filing import Filing
from .base import CRUDRepository


class FilingRepository(CRUDRepository[Filing]):
    def __init__(self) -> None:
        super().__init__(Filing)


filing_repository = FilingRepository()
