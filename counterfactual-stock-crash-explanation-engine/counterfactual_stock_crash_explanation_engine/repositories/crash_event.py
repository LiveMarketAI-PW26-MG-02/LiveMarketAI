from __future__ import annotations

from ..models.crash_event import CrashEvent
from .base import CRUDRepository


class CrashEventRepository(CRUDRepository[CrashEvent]):
    def __init__(self) -> None:
        super().__init__(CrashEvent)


crash_event_repository = CrashEventRepository()
