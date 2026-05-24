from __future__ import annotations

from ..models.whale_group import WhaleGroup
from .base import CRUDRepository


class WhaleGroupRepository(CRUDRepository[WhaleGroup]):
    def __init__(self) -> None:
        super().__init__(WhaleGroup)


whale_group_repository = WhaleGroupRepository()
