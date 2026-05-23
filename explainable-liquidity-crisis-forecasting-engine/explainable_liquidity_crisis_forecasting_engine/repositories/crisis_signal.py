from __future__ import annotations

from ..models.crisis_signal import CrisisSignal
from .base import CRUDRepository


class CrisisSignalRepository(CRUDRepository[CrisisSignal]):
    def __init__(self) -> None:
        super().__init__(CrisisSignal)


crisis_signal_repository = CrisisSignalRepository()
