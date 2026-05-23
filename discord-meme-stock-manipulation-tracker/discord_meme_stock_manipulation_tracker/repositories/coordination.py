from __future__ import annotations

from ..models.coordination import Coordination
from .base import CRUDRepository


class CoordinationRepository(CRUDRepository[Coordination]):
    def __init__(self) -> None:
        super().__init__(Coordination)


coordination_repository = CoordinationRepository()
