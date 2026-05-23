from __future__ import annotations

from ..models.sector import Sector
from .base import CRUDRepository


class SectorRepository(CRUDRepository[Sector]):
    def __init__(self) -> None:
        super().__init__(Sector)


sector_repository = SectorRepository()
