from __future__ import annotations

from ..models.snapshot import Snapshot
from .base import CRUDRepository


class SnapshotRepository(CRUDRepository[Snapshot]):
    def __init__(self) -> None:
        super().__init__(Snapshot)


snapshot_repository = SnapshotRepository()
