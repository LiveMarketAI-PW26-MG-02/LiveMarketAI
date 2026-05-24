from __future__ import annotations

from ..models.coordination_signal import CoordinationSignal
from .base import CRUDRepository


class CoordinationSignalRepository(CRUDRepository[CoordinationSignal]):
    def __init__(self) -> None:
        super().__init__(CoordinationSignal)


coordination_signal_repository = CoordinationSignalRepository()
