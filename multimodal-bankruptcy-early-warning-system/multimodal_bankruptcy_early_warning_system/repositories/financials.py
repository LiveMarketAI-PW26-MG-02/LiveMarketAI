from __future__ import annotations

from ..models.financials import Financials
from .base import CRUDRepository


class FinancialsRepository(CRUDRepository[Financials]):
    def __init__(self) -> None:
        super().__init__(Financials)


financials_repository = FinancialsRepository()
