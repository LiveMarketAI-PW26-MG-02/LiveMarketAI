from __future__ import annotations

from ..models.option import Option
from .base import CRUDRepository


class OptionRepository(CRUDRepository[Option]):
    def __init__(self) -> None:
        super().__init__(Option)


option_repository = OptionRepository()
