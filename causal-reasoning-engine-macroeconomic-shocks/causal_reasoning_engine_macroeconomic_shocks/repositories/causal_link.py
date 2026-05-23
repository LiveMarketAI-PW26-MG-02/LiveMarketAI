from __future__ import annotations

from ..models.causal_link import CausalLink
from .base import CRUDRepository


class CausalLinkRepository(CRUDRepository[CausalLink]):
    def __init__(self) -> None:
        super().__init__(CausalLink)


causal_link_repository = CausalLinkRepository()
