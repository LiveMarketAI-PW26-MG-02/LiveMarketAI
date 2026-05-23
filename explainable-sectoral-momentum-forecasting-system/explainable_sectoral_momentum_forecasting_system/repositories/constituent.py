from __future__ import annotations

from ..models.constituent import Constituent
from .base import CRUDRepository


class ConstituentRepository(CRUDRepository[Constituent]):
    def __init__(self) -> None:
        super().__init__(Constituent)


constituent_repository = ConstituentRepository()
