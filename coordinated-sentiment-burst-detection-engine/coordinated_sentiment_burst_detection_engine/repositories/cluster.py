from __future__ import annotations

from ..models.cluster import Cluster
from .base import CRUDRepository


class ClusterRepository(CRUDRepository[Cluster]):
    def __init__(self) -> None:
        super().__init__(Cluster)


cluster_repository = ClusterRepository()
