from __future__ import annotations

from ..models.explanation_report import ExplanationReport
from .base import CRUDRepository


class ExplanationReportRepository(CRUDRepository[ExplanationReport]):
    def __init__(self) -> None:
        super().__init__(ExplanationReport)


explanation_report_repository = ExplanationReportRepository()
