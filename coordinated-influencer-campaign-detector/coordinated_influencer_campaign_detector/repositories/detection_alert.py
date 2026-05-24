from __future__ import annotations

from ..models.detection_alert import DetectionAlert
from .base import CRUDRepository


class DetectionAlertRepository(CRUDRepository[DetectionAlert]):
    def __init__(self) -> None:
        super().__init__(DetectionAlert)


detection_alert_repository = DetectionAlertRepository()
