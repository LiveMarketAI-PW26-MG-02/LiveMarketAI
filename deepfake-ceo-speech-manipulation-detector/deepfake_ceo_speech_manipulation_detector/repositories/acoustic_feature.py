from __future__ import annotations

from ..models.acoustic_feature import AcousticFeature
from .base import CRUDRepository


class AcousticFeatureRepository(CRUDRepository[AcousticFeature]):
    def __init__(self) -> None:
        super().__init__(AcousticFeature)


acoustic_feature_repository = AcousticFeatureRepository()
