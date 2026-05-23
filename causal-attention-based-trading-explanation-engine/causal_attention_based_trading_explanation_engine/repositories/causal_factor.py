from __future__ import annotations

from ..models.causal_factor import CausalFactor
from .base import CRUDRepository


class CausalFactorRepository(CRUDRepository[CausalFactor]):
    def __init__(self) -> None:
        super().__init__(CausalFactor)


causal_factor_repository = CausalFactorRepository()
