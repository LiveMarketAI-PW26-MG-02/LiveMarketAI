from __future__ import annotations

from ..models.emotion_signal import EmotionSignal
from .base import CRUDRepository


class EmotionSignalRepository(CRUDRepository[EmotionSignal]):
    def __init__(self) -> None:
        super().__init__(EmotionSignal)


emotion_signal_repository = EmotionSignalRepository()
