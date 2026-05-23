from __future__ import annotations

from ..models.counterfactual import Counterfactual
from .base import CRUDRepository


class CounterfactualRepository(CRUDRepository[Counterfactual]):
    def __init__(self) -> None:
        super().__init__(Counterfactual)


counterfactual_repository = CounterfactualRepository()
