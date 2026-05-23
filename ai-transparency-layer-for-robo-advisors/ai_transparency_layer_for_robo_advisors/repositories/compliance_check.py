from __future__ import annotations

from ..models.compliance_check import ComplianceCheck
from .base import CRUDRepository


class ComplianceCheckRepository(CRUDRepository[ComplianceCheck]):
    def __init__(self) -> None:
        super().__init__(ComplianceCheck)


compliance_check_repository = ComplianceCheckRepository()
