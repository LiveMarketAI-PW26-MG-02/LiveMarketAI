from __future__ import annotations

from ..models.derivative import Derivative
from .base import CRUDRepository


class DerivativeRepository(CRUDRepository[Derivative]):
    def __init__(self) -> None:
        super().__init__(Derivative)


derivative_repository = DerivativeRepository()
