from __future__ import annotations

from ..models.decomposition import Decomposition
from .base import CRUDRepository


class DecompositionRepository(CRUDRepository[Decomposition]):
    def __init__(self) -> None:
        super().__init__(Decomposition)


decomposition_repository = DecompositionRepository()
