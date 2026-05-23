from __future__ import annotations

from ..models.whale_cluster import WhaleCluster
from .base import CRUDRepository


class WhaleClusterRepository(CRUDRepository[WhaleCluster]):
    def __init__(self) -> None:
        super().__init__(WhaleCluster)


whale_cluster_repository = WhaleClusterRepository()
