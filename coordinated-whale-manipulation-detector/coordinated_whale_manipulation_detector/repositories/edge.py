from __future__ import annotations

from ..models.edge import Edge
from .base import CRUDRepository


class EdgeRepository(CRUDRepository[Edge]):
    def __init__(self) -> None:
        super().__init__(Edge)


edge_repository = EdgeRepository()
