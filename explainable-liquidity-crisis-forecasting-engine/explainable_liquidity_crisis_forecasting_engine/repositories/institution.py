from __future__ import annotations

from ..models.institution import Institution
from .base import CRUDRepository


class InstitutionRepository(CRUDRepository[Institution]):
    def __init__(self) -> None:
        super().__init__(Institution)


institution_repository = InstitutionRepository()
