from __future__ import annotations

from ..models.shock import Shock
from .base import CRUDRepository


class ShockRepository(CRUDRepository[Shock]):
    def __init__(self) -> None:
        super().__init__(Shock)


shock_repository = ShockRepository()
