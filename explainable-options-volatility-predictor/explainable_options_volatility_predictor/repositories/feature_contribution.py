from __future__ import annotations

from ..models.feature_contribution import FeatureContribution
from .base import CRUDRepository


class FeatureContributionRepository(CRUDRepository[FeatureContribution]):
    def __init__(self) -> None:
        super().__init__(FeatureContribution)


feature_contribution_repository = FeatureContributionRepository()
