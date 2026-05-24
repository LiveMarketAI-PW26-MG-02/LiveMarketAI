from __future__ import annotations

from ..models.executive import Executive
from .base import CRUDRepository


class ExecutiveRepository(CRUDRepository[Executive]):
    def __init__(self) -> None:
        super().__init__(Executive)


executive_repository = ExecutiveRepository()
