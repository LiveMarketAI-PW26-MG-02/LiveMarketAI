from __future__ import annotations

from ..models.block import Block
from .base import CRUDRepository


class BlockRepository(CRUDRepository[Block]):
    def __init__(self) -> None:
        super().__init__(Block)


block_repository = BlockRepository()
