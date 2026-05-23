from __future__ import annotations

from ..models.rotation_signal import RotationSignal
from .base import CRUDRepository


class RotationSignalRepository(CRUDRepository[RotationSignal]):
    def __init__(self) -> None:
        super().__init__(RotationSignal)


rotation_signal_repository = RotationSignalRepository()
