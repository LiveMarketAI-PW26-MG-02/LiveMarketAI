from __future__ import annotations

from ..models.evidence import Evidence
from .base import CRUDRepository


class EvidenceRepository(CRUDRepository[Evidence]):
    def __init__(self) -> None:
        super().__init__(Evidence)


evidence_repository = EvidenceRepository()
