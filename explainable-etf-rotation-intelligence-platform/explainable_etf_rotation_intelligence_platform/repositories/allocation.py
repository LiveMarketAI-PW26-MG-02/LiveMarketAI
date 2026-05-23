from __future__ import annotations

from ..models.allocation import Allocation
from .base import CRUDRepository


class AllocationRepository(CRUDRepository[Allocation]):
    def __init__(self) -> None:
        super().__init__(Allocation)


allocation_repository = AllocationRepository()
