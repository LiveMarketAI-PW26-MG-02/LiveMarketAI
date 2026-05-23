from __future__ import annotations

from ..models.greeks import Greeks
from .base import CRUDRepository


class GreeksRepository(CRUDRepository[Greeks]):
    def __init__(self) -> None:
        super().__init__(Greeks)


greeks_repository = GreeksRepository()
