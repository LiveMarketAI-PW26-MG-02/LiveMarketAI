from __future__ import annotations

from ..models.e_t_f import ETF
from .base import CRUDRepository


class ETFRepository(CRUDRepository[ETF]):
    def __init__(self) -> None:
        super().__init__(ETF)


e_t_f_repository = ETFRepository()
