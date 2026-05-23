from __future__ import annotations

from ..models.spoof_alert import SpoofAlert
from .base import CRUDRepository


class SpoofAlertRepository(CRUDRepository[SpoofAlert]):
    def __init__(self) -> None:
        super().__init__(SpoofAlert)


spoof_alert_repository = SpoofAlertRepository()
