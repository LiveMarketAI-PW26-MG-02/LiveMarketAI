from __future__ import annotations

from ..models.insider import Insider
from .base import CRUDRepository


class InsiderRepository(CRUDRepository[Insider]):
    def __init__(self) -> None:
        super().__init__(Insider)


insider_repository = InsiderRepository()
