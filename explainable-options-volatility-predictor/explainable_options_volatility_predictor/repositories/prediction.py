from __future__ import annotations

from ..models.prediction import Prediction
from .base import CRUDRepository


class PredictionRepository(CRUDRepository[Prediction]):
    def __init__(self) -> None:
        super().__init__(Prediction)


prediction_repository = PredictionRepository()
