from __future__ import annotations

from ..models.prediction_signal import PredictionSignal
from .base import CRUDRepository


class PredictionSignalRepository(CRUDRepository[PredictionSignal]):
    def __init__(self) -> None:
        super().__init__(PredictionSignal)


prediction_signal_repository = PredictionSignalRepository()
