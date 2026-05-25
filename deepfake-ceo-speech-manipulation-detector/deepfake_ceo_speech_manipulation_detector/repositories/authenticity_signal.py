from __future__ import annotations

from ..models.authenticity_signal import AuthenticitySignal
from .base import CRUDRepository


class AuthenticitySignalRepository(CRUDRepository[AuthenticitySignal]):
    def __init__(self) -> None:
        super().__init__(AuthenticitySignal)


authenticity_signal_repository = AuthenticitySignalRepository()
