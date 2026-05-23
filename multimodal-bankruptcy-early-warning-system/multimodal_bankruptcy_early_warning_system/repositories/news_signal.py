from __future__ import annotations

from ..models.news_signal import NewsSignal
from .base import CRUDRepository


class NewsSignalRepository(CRUDRepository[NewsSignal]):
    def __init__(self) -> None:
        super().__init__(NewsSignal)


news_signal_repository = NewsSignalRepository()
