from __future__ import annotations

from ..models.advisor import Advisor
from .base import CRUDRepository


class AdvisorRepository(CRUDRepository[Advisor]):
    def __init__(self) -> None:
        super().__init__(Advisor)


advisor_repository = AdvisorRepository()
