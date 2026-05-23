from __future__ import annotations

from ..models.sentiment_signal import SentimentSignal
from .base import CRUDRepository


class SentimentSignalRepository(CRUDRepository[SentimentSignal]):
    def __init__(self) -> None:
        super().__init__(SentimentSignal)


sentiment_signal_repository = SentimentSignalRepository()
