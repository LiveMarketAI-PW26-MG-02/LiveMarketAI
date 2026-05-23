from __future__ import annotations

from ..models.dark_pool_print import DarkPoolPrint
from .base import CRUDRepository


class DarkPoolPrintRepository(CRUDRepository[DarkPoolPrint]):
    def __init__(self) -> None:
        super().__init__(DarkPoolPrint)


dark_pool_print_repository = DarkPoolPrintRepository()
