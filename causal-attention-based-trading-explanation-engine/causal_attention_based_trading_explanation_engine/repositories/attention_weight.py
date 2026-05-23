from __future__ import annotations

from ..models.attention_weight import AttentionWeight
from .base import CRUDRepository


class AttentionWeightRepository(CRUDRepository[AttentionWeight]):
    def __init__(self) -> None:
        super().__init__(AttentionWeight)


attention_weight_repository = AttentionWeightRepository()
