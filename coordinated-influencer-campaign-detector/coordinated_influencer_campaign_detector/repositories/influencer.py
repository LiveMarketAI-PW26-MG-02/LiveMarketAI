from __future__ import annotations

from ..models.influencer import Influencer
from .base import CRUDRepository


class InfluencerRepository(CRUDRepository[Influencer]):
    def __init__(self) -> None:
        super().__init__(Influencer)


influencer_repository = InfluencerRepository()
