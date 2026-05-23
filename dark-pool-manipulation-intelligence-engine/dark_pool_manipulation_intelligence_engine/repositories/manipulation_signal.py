from __future__ import annotations

from ..models.manipulation_signal import ManipulationSignal
from .base import CRUDRepository


class ManipulationSignalRepository(CRUDRepository[ManipulationSignal]):
    def __init__(self) -> None:
        super().__init__(ManipulationSignal)


manipulation_signal_repository = ManipulationSignalRepository()
