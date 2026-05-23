from __future__ import annotations

from ..models.manipulation_alert import ManipulationAlert
from .base import CRUDRepository


class ManipulationAlertRepository(CRUDRepository[ManipulationAlert]):
    def __init__(self) -> None:
        super().__init__(ManipulationAlert)


manipulation_alert_repository = ManipulationAlertRepository()
