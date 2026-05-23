from __future__ import annotations

from ..models.cohort import Cohort
from .base import CRUDRepository


class CohortRepository(CRUDRepository[Cohort]):
    def __init__(self) -> None:
        super().__init__(Cohort)


cohort_repository = CohortRepository()
