from __future__ import annotations

from ..models.rationale import Rationale
from .base import CRUDRepository


class RationaleRepository(CRUDRepository[Rationale]):
    def __init__(self) -> None:
        super().__init__(Rationale)


rationale_repository = RationaleRepository()
