from __future__ import annotations

from ..models.claim import Claim
from .base import CRUDRepository


class ClaimRepository(CRUDRepository[Claim]):
    def __init__(self) -> None:
        super().__init__(Claim)


claim_repository = ClaimRepository()
