from __future__ import annotations

from ..models.underlying import Underlying
from .base import CRUDRepository


class UnderlyingRepository(CRUDRepository[Underlying]):
    def __init__(self) -> None:
        super().__init__(Underlying)


underlying_repository = UnderlyingRepository()
