from __future__ import annotations

from ..models.burst_signal import BurstSignal
from .base import CRUDRepository


class BurstSignalRepository(CRUDRepository[BurstSignal]):
    def __init__(self) -> None:
        super().__init__(BurstSignal)


burst_signal_repository = BurstSignalRepository()
