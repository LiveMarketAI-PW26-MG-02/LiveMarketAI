from __future__ import annotations

from ..models.influence_edge import InfluenceEdge
from .base import CRUDRepository


class InfluenceEdgeRepository(CRUDRepository[InfluenceEdge]):
    def __init__(self) -> None:
        super().__init__(InfluenceEdge)


influence_edge_repository = InfluenceEdgeRepository()
